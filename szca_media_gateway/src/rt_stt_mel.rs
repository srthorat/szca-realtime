/// Streaming log-mel frontend for the cache-aware Parakeet EOU encoder.
///
/// This is a SEPARATE frontend from the one used by `rt_stt.rs`, and it has to
/// be. The full-utterance path feeds `parakeet_nemo128.onnx`, an ONNX graph whose
/// `nemo_preprocessor` function ends in a `normalize` sub-function applying
/// **per-utterance** mean/variance normalization. That is correct when you hold
/// the whole utterance, and wrong for streaming: normalizing each 1.28 s chunk by
/// its own statistics makes the features non-stationary across chunk boundaries.
/// Verified against the real model — with per-feature (or any) normalization the
/// EOU decoder emits **zero tokens**; with raw log-mel it decodes correctly:
///
/// ```text
/// per_feature normalization -> ''             (0 tokens)
/// global mean/var           -> ''             (0 tokens)
/// none (raw log-mel)        -> 'hello world'  ✅
/// ```
///
/// So the streaming contract is `log(mel + 2^-24)` and nothing else. Every
/// constant below was pinned by running the real graph, not read off a card:
///
/// | Parameter | Value |
/// |-----------|-------|
/// | mel bins | 128 |
/// | n_fft | 512 |
/// | hop | 160 (10 ms @ 16 kHz) |
/// | win_length | 400 (25 ms), centered inside the 512-pt FFT frame |
/// | window | periodic Hann |
/// | padding | reflect, `n_fft / 2` both sides |
/// | filterbank | Slaney-normalized triangular, HTK mel scale |
/// | log | `ln(x + 2^-24)` |
/// | normalization | **none** |
///
/// The log guard matters: `1e-5` truncates `"hello world"` to `"hello"`.
/// Pre-emphasis (0.97, per the model's `config.json`) made no measurable
/// difference to decoded text but is applied for faithfulness to the export.
use std::f32::consts::PI;

use realfft::{RealFftPlanner, RealToComplex};
use std::sync::Arc;

/// Number of mel filterbank channels the encoder expects.
pub const N_MELS: usize = 128;

/// FFT size.
pub const N_FFT: usize = 512;

/// Hop length in samples (10 ms at 16 kHz).
pub const HOP: usize = 160;

/// Analysis window length in samples (25 ms at 16 kHz).
pub const WIN: usize = 400;

/// Sample rate the frontend (and the model) assumes.
pub const SAMPLE_RATE: u32 = 16_000;

/// Pre-emphasis coefficient from the checkpoint's `config.json`.
const PREEMPH: f32 = 0.97;

/// Additive guard inside the log. This is `2^-24`, matching the NeMo export.
/// A larger guard (e.g. `1e-5`) measurably truncates transcripts.
const LOG_GUARD: f32 = 5.960_464_5e-8;

/// Number of frequency bins in a real FFT of size `N_FFT`.
const N_BINS: usize = N_FFT / 2 + 1;

/// HTK mel scale (the 1127·ln form NeMo uses).
fn hz_to_mel(f: f32) -> f32 {
    1127.0 * (1.0 + f / 700.0).ln()
}

/// Inverse of [`hz_to_mel`].
fn mel_to_hz(m: f32) -> f32 {
    700.0 * ((m / 1127.0).exp() - 1.0)
}

/// Build the Slaney-normalized triangular mel filterbank, `[N_MELS][N_BINS]`.
///
/// "Slaney" here means each triangle is scaled by `2 / (hi_hz - lo_hz)` so
/// filters have unit *area* rather than unit peak. Dropping that normalization
/// changes the log-mel magnitudes and therefore the encoder's input
/// distribution.
fn mel_filterbank() -> Vec<Vec<f32>> {
    let f_max = SAMPLE_RATE as f32 / 2.0;
    let mel_lo = hz_to_mel(0.0);
    let mel_hi = hz_to_mel(f_max);

    // N_MELS + 2 mel-spaced points: each filter spans a (lo, center, hi) triple.
    let hz_pts: Vec<f32> = (0..N_MELS + 2)
        .map(|i| {
            let m = mel_lo + (mel_hi - mel_lo) * (i as f32) / (N_MELS + 1) as f32;
            mel_to_hz(m)
        })
        .collect();

    // Linear FFT bin center frequencies.
    let bin_hz: Vec<f32> = (0..N_BINS)
        .map(|b| (b as f32) * f_max / ((N_BINS - 1) as f32))
        .collect();

    let mut fb = vec![vec![0.0_f32; N_BINS]; N_MELS];
    for m in 0..N_MELS {
        let (lo, ctr, hi) = (hz_pts[m], hz_pts[m + 1], hz_pts[m + 2]);
        // Slaney area normalization.
        let enorm = 2.0 / (hz_pts[m + 2] - hz_pts[m]).max(1e-10);
        for (b, &f) in bin_hz.iter().enumerate() {
            let left = (f - lo) / (ctr - lo).max(1e-10);
            let right = (hi - f) / (hi - ctr).max(1e-10);
            let w = left.min(right).max(0.0);
            fb[m][b] = w * enorm;
        }
    }
    fb
}

/// Periodic Hann window of length `WIN` (`hann(N+1)[:N]`, not symmetric).
fn hann_periodic() -> Vec<f32> {
    (0..WIN)
        .map(|i| 0.5 - 0.5 * (2.0 * PI * i as f32 / WIN as f32).cos())
        .collect()
}

/// Reusable log-mel extractor. Holds the FFT plan, window and filterbank so the
/// hot path allocates nothing per frame.
pub struct MelFrontend {
    fft: Arc<dyn RealToComplex<f32>>,
    window: Vec<f32>,
    fb: Vec<Vec<f32>>,
    // Scratch reused across frames.
    frame: Vec<f32>,
    spectrum: Vec<realfft::num_complex::Complex<f32>>,
    power: Vec<f32>,
}

impl MelFrontend {
    /// Build the frontend (plans the FFT, precomputes window + filterbank).
    pub fn new() -> Self {
        let mut planner = RealFftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(N_FFT);
        let spectrum = fft.make_output_vec();
        Self {
            fft,
            window: hann_periodic(),
            fb: mel_filterbank(),
            frame: vec![0.0; N_FFT],
            spectrum,
            power: vec![0.0; N_BINS],
        }
    }

    /// Compute log-mel features for `samples` (f32 in [-1, 1], 16 kHz mono).
    ///
    /// Returns features in **frame-major** order: `frames[t][mel]`. The caller
    /// transposes into the `[1, 128, T]` layout the encoder wants.
    ///
    /// `samples` is treated as a complete signal: it is reflect-padded by
    /// `N_FFT / 2` on both sides, matching a "centered" STFT.
    pub fn log_mel(&mut self, samples: &[f32]) -> Vec<[f32; N_MELS]> {
        if samples.is_empty() {
            return Vec::new();
        }

        // Pre-emphasis: y[0] = x[0], y[n] = x[n] - a*x[n-1].
        let mut pre = Vec::with_capacity(samples.len());
        pre.push(samples[0]);
        for i in 1..samples.len() {
            pre.push(samples[i] - PREEMPH * samples[i - 1]);
        }

        // Reflect-pad N_FFT/2 both sides. `reflect` excludes the edge sample
        // itself (numpy's default), so index 1..=pad mirrors around index 0.
        let pad = N_FFT / 2;
        let mut x = Vec::with_capacity(pre.len() + 2 * pad);
        for i in (1..=pad).rev() {
            x.push(pre[i.min(pre.len().saturating_sub(1))]);
        }
        x.extend_from_slice(&pre);
        for i in 1..=pad {
            let idx = pre.len().saturating_sub(1 + i);
            x.push(pre[idx]);
        }

        if x.len() < N_FFT {
            return Vec::new();
        }
        let n_frames = 1 + (x.len() - N_FFT) / HOP;
        let mut out = Vec::with_capacity(n_frames);

        // win_length (400) < n_fft (512): the window sits CENTERED in the FFT
        // frame with zeros either side, so samples outside [off, off+WIN) are
        // dropped rather than rectangular-windowed.
        let off = (N_FFT - WIN) / 2;

        for t in 0..n_frames {
            let seg = &x[t * HOP..t * HOP + N_FFT];
            self.frame.fill(0.0);
            for i in 0..WIN {
                self.frame[off + i] = seg[off + i] * self.window[i];
            }

            // Power spectrum |X|^2.
            if self
                .fft
                .process(&mut self.frame, &mut self.spectrum)
                .is_err()
            {
                return out;
            }
            for (p, c) in self.power.iter_mut().zip(self.spectrum.iter()) {
                *p = c.re * c.re + c.im * c.im;
            }

            // Mel projection then log. NO normalization — see the module docs.
            let mut feats = [0.0_f32; N_MELS];
            for (m, row) in self.fb.iter().enumerate() {
                let mut acc = 0.0_f32;
                for (b, &w) in row.iter().enumerate() {
                    if w != 0.0 {
                        acc += w * self.power[b];
                    }
                }
                feats[m] = (acc + LOG_GUARD).ln();
            }
            out.push(feats);
        }
        out
    }
}

impl Default for MelFrontend {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Golden values captured from the Python reference implementation that was
    /// verified to decode "hello world" through the real EOU model. These pin the
    /// filterbank shape: a silent regression here means silent transcript loss.
    #[test]
    fn filterbank_matches_reference() {
        let fb = mel_filterbank();
        assert_eq!(fb.len(), N_MELS);
        assert_eq!(fb[0].len(), N_BINS);

        let total: f32 = fb.iter().map(|r| r.iter().sum::<f32>()).sum();
        assert!(
            (total - 4.075_746).abs() < 1e-3,
            "filterbank total sum {total} != reference 4.075746"
        );

        // Row 0's triangle falls between FFT bins, so it collects no energy —
        // this is expected for 128 mels at 16 kHz and matches the reference.
        assert_eq!(fb[0].iter().sum::<f32>(), 0.0);
        for (row, expect) in [(1usize, 0.053_867_8_f32), (64, 0.030_574_2), (127, 0.031_859_5)] {
            let got: f32 = fb[row].iter().sum();
            assert!(
                (got - expect).abs() < 1e-4,
                "fb row {row} sum {got} != reference {expect}"
            );
        }
    }

    #[test]
    fn hann_is_periodic_not_symmetric() {
        let w = hann_periodic();
        assert_eq!(w.len(), WIN);
        assert!(w[0].abs() < 1e-6);
        // A periodic Hann never returns to exactly 0 at the last sample (a
        // symmetric one does), and it is symmetric about index 0 rather than
        // about the window centre, so w[N-1] == w[1].
        assert!(w[WIN - 1] > 0.0, "window looks symmetric, not periodic");
        assert!(
            (w[WIN - 1] - w[1]).abs() < 1e-9,
            "w[N-1] {} != w[1] {} — not a periodic Hann",
            w[WIN - 1],
            w[1]
        );
    }

    #[test]
    fn mel_scale_roundtrip() {
        for hz in [0.0_f32, 100.0, 1000.0, 8000.0] {
            let back = mel_to_hz(hz_to_mel(hz));
            assert!((back - hz).abs() < 1e-2, "{hz} -> {back}");
        }
    }

    /// Frame count and value distribution against the verified reference for a
    /// deterministic 440 Hz + 1 kHz mixture (0.5 s at 16 kHz).
    #[test]
    fn log_mel_matches_reference_stats() {
        let n = 8000;
        let syn: Vec<f32> = (0..n)
            .map(|i| {
                let t = i as f32 / SAMPLE_RATE as f32;
                0.5 * (2.0 * PI * 440.0 * t).sin() + 0.25 * (2.0 * PI * 1000.0 * t).sin()
            })
            .collect();

        let mut fe = MelFrontend::new();
        let feats = fe.log_mel(&syn);

        // Reference: shape [128, 51] (mel-major) => 51 frames here.
        assert_eq!(feats.len(), 51, "frame count mismatch vs reference");

        let flat: Vec<f32> = feats.iter().flat_map(|f| f.iter().copied()).collect();
        let mean = flat.iter().sum::<f32>() / flat.len() as f32;
        let var = flat.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / flat.len() as f32;
        let std = var.sqrt();

        // Reference: mean -13.1475, std 5.0721, min -16.6355, max 1.0418.
        assert!((mean + 13.147_55).abs() < 0.05, "mean {mean} != -13.1475");
        assert!((std - 5.072_1).abs() < 0.05, "std {std} != 5.0721");

        let min = flat.iter().cloned().fold(f32::INFINITY, f32::min);
        let max = flat.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        assert!((min + 16.635_53).abs() < 0.05, "min {min} != -16.6355");
        assert!((max - 1.041_79).abs() < 0.05, "max {max} != 1.0418");

        // Spot-check an individual frame against the reference vector.
        let ref_f10_mel0_8 = [
            -16.635_532_f32, -16.027_416, -16.411_812, -15.090_261, -15.479_733, -14.711_424,
            -14.957_273, -15.429_456,
        ];
        for (i, &want) in ref_f10_mel0_8.iter().enumerate() {
            let got = feats[10][i];
            assert!(
                (got - want).abs() < 0.05,
                "frame10 mel{i}: {got} != reference {want}"
            );
        }
    }

    #[test]
    fn log_mel_no_normalization_applied() {
        // A normalized frontend would produce ~zero mean / unit std. Raw log-mel
        // must NOT: this test is what stops someone "helpfully" adding it back.
        let mut fe = MelFrontend::new();
        let quiet: Vec<f32> = (0..8000).map(|i| 0.01 * (i as f32 * 0.1).sin()).collect();
        let feats = fe.log_mel(&quiet);
        let flat: Vec<f32> = feats.iter().flat_map(|f| f.iter().copied()).collect();
        let mean = flat.iter().sum::<f32>() / flat.len() as f32;
        assert!(
            mean < -5.0,
            "mean {mean} looks normalized; the EOU decoder emits 0 tokens if it is"
        );
    }

    #[test]
    fn log_mel_empty_and_short_input() {
        let mut fe = MelFrontend::new();
        assert!(fe.log_mel(&[]).is_empty());
        // Shorter than one padded frame still must not panic.
        let _ = fe.log_mel(&[0.1, -0.1, 0.2]);
    }
}
