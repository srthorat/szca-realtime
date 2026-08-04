#![allow(
    clippy::needless_range_loop,
    clippy::assign_op_pattern,
    clippy::needless_late_init,
    clippy::manual_div_ceil
)]

/// Real-time DeepFilterNet3 streaming DSP: PCM → STFT → ERB/DF features →
/// Dfn3Model inference → mask/coefs → iSTFT → enhanced PCM.
///
/// This module bridges our existing [`Dfn3Model`] (ONNX inference via `ort`)
/// with the `deep_filter` crate's low-level transforms. It handles the full
/// 16 kHz ↔ 48 kHz resampling, STFT framing, feature extraction, gain+filter
/// application, and streaming state across chunks.
///
/// ## Processing chain (per block of S frames at 48 kHz)
///
/// ```text
/// 16kHz PCM ──┬─► resample 16→48 kHz
///             │
///             ▼
///     DFState::analysis() per frame → complex spectrum [481 bins]
///             │
///             ├─► compute_band_corr + band_compr → 32 ERB log-power
///             │       └─► band_mean_norm_erb (running) → feat_erb [1,1,S,32]
///             │
///             ├─► first 96 complex bins
///             │       └─► band_unit_norm (running) → feat_spec [1,2,S,96]
///             │
///             ▼
///     Dfn3Model::run(S, feat_erb, feat_spec)
///             │
///             ├─► ERB mask → apply_interp_band_gain → full spectrum
///             └─► DF coefs → deep filter convolution (5-tap per band)
///                         → enhanced spectrum (first 96 bins)
///             │
///             ▼
///     DFState::synthesis() per frame → 48 kHz PCM
///             │
///             └─► resample 48→16 kHz → 16 kHz PCM
/// ```

use df::{Complex32, DFState, MEAN_NORM_INIT, UNIT_NORM_INIT};

use crate::dfn3::{Dfn3Model, DFN3_NB_DF, DFN3_NB_ERB, DFN3_FFT, DFN3_HOP, DFN3_SR};

// ---------------------------------------------------------------------------
// Constants (from dfn3_config.ini [df] section)
// ---------------------------------------------------------------------------

/// FFT frequency bins: fft_size/2 + 1.
const FREQ_BINS: usize = DFN3_FFT / 2 + 1; // 481

/// Internal resampler ratio (48 kHz → 16 kHz).
const RESAMPLE_RATIO: usize = (DFN3_SR / 16000) as usize; // 3

/// Running-norm alpha computed from norm_tau=1 and hop=480 @ 48 kHz.
/// alpha = exp(-hop / (sr * tau)) = exp(-480 / (48000 * 1)).
const NORM_ALPHA: f32 = 0.99005_f32;

/// Number of deep-filter taps.
const DF_ORDER: usize = 5;

/// Deep-filter lookahead frames.
const DF_LOOKAHEAD: usize = 2;

/// Internal processing block size (frames). Must be >= DF_ORDER so the deep
/// filter has enough context. 8 frames = 8 * 480 / 48000 = 80ms of 48 kHz
/// audio ≈ 53ms of 16 kHz input. Everything before the lookahead boundary
/// and after the effective end is discarded (output shrinks by DF_LOOKAHEAD
/// frames at each end).
const BLOCK_FRAMES: usize = 8;

/// Output frames per block after trimming lookahead margin at both ends.
const VALID_FRAMES: usize = BLOCK_FRAMES - 2 * DF_LOOKAHEAD; // 4 frames

/// Resample state for 16↔48 kHz streaming via linear interpolation.
struct LinearResampler {
    /// Accumulated input samples not yet consumed.
    buf: Vec<f32>,
    /// Fractional position within resample ratio.
    phase: usize,
}

impl LinearResampler {
    fn new() -> Self {
        Self {
            buf: Vec::new(),
            phase: 0,
        }
    }

    /// Push incoming 16 kHz samples. Returns 48 kHz output when enough input
    /// is buffered; caller should call [`take_output`] afterwards.
    fn push(&mut self, samples: &[f32]) {
        self.buf.extend_from_slice(samples);
    }

    /// Drain as many complete 48 kHz frames as possible. Each output sample
    /// is linearly interpolated from the nearest two 16 kHz input samples.
    /// Returns `Some((48kHz_samples, consumed_16kHz_samples))` or `None` if
    /// not enough input to produce at least one 48 kHz sample.
    fn produce_next(&mut self, output_len: usize) -> Option<(Vec<f32>, usize)> {
        // We need ceil(output_len * 16/48) = output_len / 3 input samples.
        let needed_input = (output_len * 16000 + DFN3_SR as usize - 1) / DFN3_SR as usize;
        if self.buf.len() < needed_input {
            return None;
        }
        let mut out = Vec::with_capacity(output_len);
        let consumed: usize;
        let mut idx = self.phase as f32;
        for _ in 0..output_len {
            let lo = idx as usize;
            let hi = (lo + 1).min(self.buf.len() - 1);
            let frac = idx - lo as f32;
            let v = self.buf[lo] * (1.0 - frac) + self.buf[hi] * frac;
            out.push(v);
            idx += 1.0 / RESAMPLE_RATIO as f32;
        }
        consumed = needed_input;
        // Remove consumed samples from buffer.
        self.buf.drain(0..consumed);
        // Update phase for fractional remainder.
        let total_idx = idx;
        self.phase = (total_idx * RESAMPLE_RATIO as f32) as usize % RESAMPLE_RATIO;
        Some((out, consumed))
    }

    /// Down-resample 48 kHz → 16 kHz: decimate by RESAMPLE_RATIO with
    /// simple averaging filter to avoid aliasing.
    fn down(&self, input_48k: &[f32]) -> Vec<f32> {
        let out_len = input_48k.len() / RESAMPLE_RATIO;
        let mut out = Vec::with_capacity(out_len);
        for chunk in input_48k.chunks(RESAMPLE_RATIO) {
            let sum: f32 = chunk.iter().sum();
            out.push(sum / chunk.len() as f32);
        }
        out
    }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Real-time DFN3 streaming DSP engine.
///
/// Owns:
/// - The three ONNX sessions via [`Dfn3Model`]
/// - STFT/iSTFT state via [`DFState`] (the `df` crate)
/// - Running normalization states for ERB and DF features
/// - Resample buffer for 16↔48 kHz conversion
/// - Rolling spectrum buffer for deep filter lookahead/lookback
pub struct Dfn3Dsp {
    model: Dfn3Model,
    stft: DFState,
    istft: DFState,
    /// Running mean state for ERB features: `[nb_erb]` f32.
    erb_norm_state: Vec<f32>,
    /// Running unit-norm state for DF features: `[nb_df]` f32.
    df_norm_state: Vec<f32>,
    /// Rolling complex-spectrum ring buffer for deep filter: `[df_order][freq_bins]`.
    spec_ring: Vec<Vec<Complex32>>,
    /// Write index into `spec_ring`.
    ring_pos: usize,
    /// Resampler state.
    resampler: LinearResampler,
    /// Accumulated 48 kHz output samples between calls.
    out_buf_48k: Vec<f32>,
    /// Number of 48 kHz frames sent to the NN so far.
    total_frames: usize,
}

impl Dfn3Dsp {
    /// Load the DFN3 model and initialise all DSP state.
    pub fn load(model: Dfn3Model) -> Self {
        let stft = DFState::new(
            DFN3_SR as usize,
            DFN3_FFT,
            DFN3_HOP,
            DFN3_NB_ERB,
            2, // min_nb_erb_freqs
        );
        // iSTFT needs its own DFState (separate overlap buffers).
        let istft = DFState::new(
            DFN3_SR as usize,
            DFN3_FFT,
            DFN3_HOP,
            DFN3_NB_ERB,
            2,
        );

        // Initialise running-norm accumulators.
        let mut erb_norm_state = vec![0.0_f32; DFN3_NB_ERB];
        // ERB mean norm: initial state from MEAN_NORM_INIT, linearly spaced.
        if DFN3_NB_ERB > 1 {
            let step = (MEAN_NORM_INIT[1] - MEAN_NORM_INIT[0]) / (DFN3_NB_ERB - 1) as f32;
            for (i, s) in erb_norm_state.iter_mut().enumerate() {
                *s = MEAN_NORM_INIT[0] + i as f32 * step;
            }
        } else {
            erb_norm_state[0] = MEAN_NORM_INIT[0];
        }

        let df_norm_state = vec![UNIT_NORM_INIT[0]; DFN3_NB_DF];

        let spec_ring = vec![vec![Complex32::default(); FREQ_BINS]; DF_ORDER];

        Self {
            model,
            stft,
            istft,
            erb_norm_state,
            df_norm_state,
            spec_ring,
            ring_pos: 0,
            resampler: LinearResampler::new(),
            out_buf_48k: Vec::new(),
            total_frames: 0,
        }
    }

    /// Process one block of incoming 16 kHz PCM audio through the DFN3 chain.
    ///
    /// Returns enhanced 16 kHz PCM. May return empty if buffering hasn't
    /// accumulated enough audio for a full block yet.
    pub fn process(&mut self, pcm_16k: &[i16]) -> Vec<i16> {
        // Convert to f32.
        let samples_f32: Vec<f32> = pcm_16k.iter().map(|&s| s as f32 / 32768.0).collect();
        self.resampler.push(&samples_f32);

        // Try to drain a full block's worth of 48 kHz input.
        let block_samples_48k = DFN3_HOP * BLOCK_FRAMES; // 8 * 480 = 3840
        while let Some((frame_48k, _consumed)) = self.resampler.produce_next(block_samples_48k) {
            self.process_block_48k(&frame_48k);
        }

        // Drain any available 48 kHz output and down-resample.
        let out_48k: Vec<f32> = self.out_buf_48k.drain(..).collect();
        if out_48k.is_empty() {
            return Vec::new();
        }
        let out_16k = self.resampler.down(&out_48k);
        out_16k.iter().map(|&v| (v * 32768.0).clamp(-32768.0, 32767.0) as i16).collect()
    }

    /// Process a full block of 48 kHz PCM (exactly `BLOCK_FRAMES × HOP`
    /// samples) through the STFT → features → NN → mask → iSTFT chain.
    fn process_block_48k(&mut self, pcm_48k: &[f32]) {
        assert_eq!(pcm_48k.len(), DFN3_HOP * BLOCK_FRAMES);

        // ---- 1. STFT: complex spectrum per frame ----
        // Store as [BLOCK_FRAMES][freq_bins] Complex32.
        let mut spectra: Vec<Vec<Complex32>> = Vec::with_capacity(BLOCK_FRAMES);
        for frame_i in 0..BLOCK_FRAMES {
            let start = frame_i * DFN3_HOP;
            let end = start + DFN3_HOP;
            // Build 2D input of shape [1, HOP] for DFState::analysis.
            // analysis expects &[f32] (mono interleaved).
            let chunk = &pcm_48k[start..end];
            let mut spec_frame = vec![Complex32::default(); FREQ_BINS];
            self.stft.analysis(chunk, &mut spec_frame);
            spectra.push(spec_frame);
        }

        // ---- 2. Extract ERB and DF features for each frame ----
        let mut erb_feats = vec![0.0_f32; BLOCK_FRAMES * DFN3_NB_ERB];
        let mut df_feats = vec![0.0_f32; 2 * BLOCK_FRAMES * DFN3_NB_DF];

        for (t, spec) in spectra.iter().enumerate() {
            // ERB features: complex correlation → band compression → mean norm
            let mag_spec: Vec<f32> = spec.iter().map(|c| c.norm()).collect();
            // Compress: convert to dB power via per-band averaging
            let mut erb_log = vec![0.0_f32; DFN3_NB_ERB];
            band_compr(&mut erb_log, &mag_spec, &self.stft.erb);
            // Convert to dB scale
            for v in erb_log.iter_mut() {
                *v = (v.abs() + 1e-10).log10() * 10.0;
            }
            // Running mean normalization
            band_mean_norm_erb(&mut erb_log, &mut self.erb_norm_state, NORM_ALPHA);
            // Write to feature buffer (S × nb_erb = S * 32)
            for b in 0..DFN3_NB_ERB {
                erb_feats[t * DFN3_NB_ERB + b] = erb_log[b];
            }

            // DF features: first 96 complex bins → unit norm → [real, imag]
            let mut cplx_df: Vec<Complex32> = spec[..DFN3_NB_DF].to_vec();
            band_unit_norm(&mut cplx_df, &mut self.df_norm_state, NORM_ALPHA);
            // Write real and imag into feat_spec [2 × S × nb_df]
            for b in 0..DFN3_NB_DF {
                df_feats[t * DFN3_NB_DF + b] = cplx_df[b].re; // real block
                df_feats[BLOCK_FRAMES * DFN3_NB_DF + t * DFN3_NB_DF + b] = cplx_df[b].im; // imag block
            }
        }

        // ---- 3. Run the Dfn3Model neural network ----
        let model = &mut self.model;
        let out = match model.run_flat(BLOCK_FRAMES, &erb_feats, &df_feats) {
            Ok(o) => o,
            Err(e) => {
                tracing::error!(error = %e, "DFN3 model inference failed");
                // Fall through: output whatever we have (may be silence or
                // original audio if processing is disabled).
                return;
            }
        };

        // ---- 4. Apply ERB mask to full spectrum ----
        // out.mask length = BLOCK_FRAMES * DFN3_NB_ERB
        // Apply per-frame gains to each frame's spectrum
        for t in 0..BLOCK_FRAMES {
            let mask_start = t * DFN3_NB_ERB;
            let mask = &out.mask[mask_start..mask_start + DFN3_NB_ERB];
            apply_band_gain(&mut spectra[t], mask, &self.stft.erb);
        }

        // ---- 5. Update rolling spectrum ring buffer ----
        for t in 0..BLOCK_FRAMES {
            self.spec_ring[self.ring_pos] = spectra[t].clone();
            self.ring_pos = (self.ring_pos + 1) % DF_ORDER;
        }

        // ---- 6. Apply deep-filter coefficients to first nb_df bins ----
        // out.coefs layout: for each frame t, [nb_df * df_order * 2] values
        // representing [real_coef_0_0, imag_coef_0_0, real_coef_1_0, ...]
        // We interpret it as: for frame t, coef[f][k][c] where c=0=real, c=1=imag
        let coefs_per_frame = DFN3_NB_DF * DF_ORDER * 2;
        if out.coefs.len() >= BLOCK_FRAMES * coefs_per_frame {
            for t in 0..BLOCK_FRAMES {
                let cf_start = t * coefs_per_frame;
                // Build complex coefs array: [nb_df][df_order] Complex32
                let mut coefs_2d: Vec<Vec<Complex32>> = vec![vec![Complex32::default(); DF_ORDER]; DFN3_NB_DF];
                for b in 0..DFN3_NB_DF {
                    for k in 0..DF_ORDER {
                        let base = cf_start + (b * DF_ORDER + k) * 2;
                        coefs_2d[b][k] = Complex32::new(out.coefs[base], out.coefs[base + 1]);
                    }
                }
                // Convolve with spectrum history
                let mut enhanced = vec![Complex32::default(); DFN3_NB_DF];
                for b in 0..DFN3_NB_DF {
                    for k in 0..DF_ORDER {
                        // ring index for frame t - k (commented out - was dead code that panicked on overflow)
                        // Offset: we want spec at position (t - k + DF_LOOKAHEAD) relative to current processing
                        // Since we're processing BLOCK_FRAMES at once, the ring buffer has all the history.
                        // We need to handle frame indices carefully.
                        // Simple approach: for frame t within the block, find the spectrum from k frames ago.
                        // spectra[t - k + DF_LOOKAHEAD] if within 0..t, else use the ring buffer
                        let s = if t >= k {
                            spectra[t - k][b]
                        } else if k - t <= self.total_frames {
                            // Use ring buffer for frames before the current block
                            self.spec_ring[(self.ring_pos + k - t) % DF_ORDER][b]
                        } else {
                            Complex32::default()
                        };
                        enhanced[b] = enhanced[b] + s * coefs_2d[b][k];
                    }
                }
                // Write enhanced first nb_df bins back (remaining bins already
                // have ERB gain from step 4).
                spectra[t][..DFN3_NB_DF].copy_from_slice(&enhanced[..DFN3_NB_DF]);
            }
        }
        self.total_frames += BLOCK_FRAMES;

        // ---- 7. iSTFT: complex spectrum back to 48 kHz PCM ----
        let mut output_48k = vec![0.0_f32; DFN3_HOP * VALID_FRAMES];
        // Only keep the VALID_FRAMES at the center (trim lookahead margin).
        for t in 0..VALID_FRAMES {
            let src_t = t + DF_LOOKAHEAD; // skip first DF_LOOKAHEAD frames
            let start = t * DFN3_HOP;
            let end = start + DFN3_HOP;
            let chunk = &mut output_48k[start..end];
            self.istft.synthesis(&mut spectra[src_t], chunk);
        }
        self.out_buf_48k.extend_from_slice(&output_48k);
    }
}

// ---------------------------------------------------------------------------
// Feature-extraction helper functions (mirror df crate's internal functions)
// ---------------------------------------------------------------------------

/// Band-correlation feature (project complex spectrum onto ERB bands).
#[allow(dead_code)]
fn compute_band_corr(out: &mut [f32], x: &[Complex32], p: &[Complex32], erb_fb: &[usize]) {
    for y in out.iter_mut() {
        *y = 0.0;
    }
    let mut bcsum = 0;
    for (&band_size, out_b) in erb_fb.iter().zip(out.iter_mut()) {
        let k = 1.0 / band_size as f32;
        for j in 0..band_size {
            let idx = bcsum + j;
            *out_b += (x[idx].re * p[idx].re + x[idx].im * p[idx].im) * k;
        }
        bcsum += band_size;
    }
}

/// Band average (mean per ERB band).
fn band_compr(out: &mut [f32], x: &[f32], erb_fb: &[usize]) {
    for y in out.iter_mut() {
        *y = 0.0;
    }
    let mut bcsum = 0;
    for (&band_size, out_b) in erb_fb.iter().zip(out.iter_mut()) {
        let k = 1.0 / band_size as f32;
        for j in 0..band_size {
            let idx = bcsum + j;
            *out_b += x[idx] * k;
        }
        bcsum += band_size;
    }
}

/// Running mean normalization for ERB-band features.
fn band_mean_norm_erb(xs: &mut [f32], state: &mut [f32], alpha: f32) {
    for (x, s) in xs.iter_mut().zip(state.iter_mut()) {
        *s = *x * (1.0 - alpha) + *s * alpha;
        *x -= *s;
        *x /= 40.0;
    }
}

/// Running unit normalization for complex features.
fn band_unit_norm(xs: &mut [Complex32], state: &mut [f32], alpha: f32) {
    for (x, s) in xs.iter_mut().zip(state.iter_mut()) {
        *s = x.norm() * (1.0 - alpha) + *s * alpha;
        *x /= s.sqrt().max(1e-10);
    }
}

/// Apply ERB band gains to complex spectrum via interpolation (same gain
/// value applied to all FFT bins within each ERB band).
fn apply_band_gain(spec: &mut [Complex32], band_gains: &[f32], erb_fb: &[usize]) {
    let mut bcsum = 0;
    for (&band_size, &gain) in erb_fb.iter().zip(band_gains.iter()) {
        for j in 0..band_size {
            let idx = bcsum + j;
            if idx < spec.len() {
                spec[idx] *= gain;
            }
        }
        bcsum += band_size;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear_resampler_up() {
        let mut r = LinearResampler::new();
        // 16 kHz: 320 samples = 20ms
        let inp: Vec<f32> = (0..320).map(|i| (i as f32) / 320.0).collect();
        r.push(&inp);
        // 320 input samples @ 16kHz = 960 output samples @ 48kHz
        let result = r.produce_next(960);
        assert!(result.is_some(), "should produce output from 320 input");
        let (out, consumed) = result.unwrap();
        assert_eq!(out.len(), 960);
        assert!(consumed >= 320 && consumed <= 640, "should consume ~320 input samples, got {consumed}");
        // Output should roughly track input (scaled by 3×)
        assert!(out[0] >= 0.0 && out[0] <= 1.0, "out[0]={} out of range", out[0]);
    }

    #[test]
    fn test_linear_resampler_down() {
        let r = LinearResampler::new();
        // 48 kHz: 960 samples = 20ms
        let inp_48k: Vec<f32> = (0..960).map(|i| ((i % 3) as f32) / 3.0).collect();
        let out_16k = r.down(&inp_48k);
        assert_eq!(out_16k.len(), 320);
        // Even a simple decimate with average should preserve the signal shape
        assert!(out_16k[0] >= 0.0 && out_16k[0] <= 1.0);
    }

    #[test]
    fn test_erb_feature_extraction() {
        // Verify that ERB features produce valid outputs from random input
        let s = DFState::new(48000, 960, 480, 32, 2);
        let spec_len = 481;
        // Random complex spectrum
        let spec: Vec<Complex32> = (0..spec_len)
            .map(|i| Complex32::new(0.01 * (i % 7) as f32, 0.01 * ((i % 3) as f32 - 1.5)))
            .collect();

        let mag_spec: Vec<f32> = spec.iter().map(|c| c.norm()).collect();
        let mut erb_log = vec![0.0_f32; 32];
        band_compr(&mut erb_log, &mag_spec, &s.erb);
        for v in erb_log.iter_mut() {
            *v = (v.abs() + 1e-10).log10() * 10.0;
        }
        assert!(erb_log.iter().all(|v| v.is_finite()), "ERB log must be finite");

        // After norm
        let mut state = vec![-60.0_f32; 32];
        band_mean_norm_erb(&mut erb_log, &mut state, 0.99);
        assert!(erb_log.iter().all(|v| v.is_finite()));
        // State should have advanced
        assert!(state[0].is_finite() && state[0] < 0.0);
    }

    #[test]
    fn test_apply_band_gain() {
        let s = DFState::new(48000, 960, 480, 32, 2);
        let mut spec: Vec<Complex32> = (0..481).map(|i| Complex32::new(i as f32, 0.0)).collect();
        let gains = vec![0.5_f32; 32];
        let original = spec[10];
        apply_band_gain(&mut spec, &gains, &s.erb);
        assert!((spec[10].re - original.re * 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_df_norm_runs() {
        let mut cplx: Vec<Complex32> = (0..96)
            .map(|i| Complex32::new((i % 7) as f32 * 0.1, ((i % 5) as f32 - 2.0) * 0.1))
            .collect();
        let mut state = vec![UNIT_NORM_INIT[0]; 96];
        band_unit_norm(&mut cplx, &mut state, 0.99);
        assert!(cplx.iter().all(|v| v.norm().is_finite()));
        assert!(state.iter().all(|v| v.is_finite() && *v > 0.0));
    }
}
