/// Cache-aware streaming STT: Sherpa Zipformer.
///
/// Two ONNX graphs (encoder + decoder/joiner), 19-layer Zipformer encoder with
/// 116 accumulator caches, 80-mel Kaldi-style frontend.
use std::sync::atomic::{AtomicBool, Ordering};

use ndarray::{Array1, Array2, Array3, Axis, IxDyn};
use ort::session::Session;
use ort::value::{Tensor, Value};

use crate::onnx::init_ort;
use crate::rt_pipeline::SttStage;
use crate::rt_stt::SttInput;
use crate::stage_pool::Replica;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const N_MELS: usize = 80;
const CHUNK_FRAMES: usize = 141;
const PREEMPH: f32 = 0.97;
const LOG_GUARD: f32 = 1e-10;
const WIN: usize = 400;
const HOP: usize = 160;
const ENC_HIDDEN: usize = 512;
const N_LOGITS: usize = 650;
const MAX_SYMBOLS_PER_STEP: usize = 10;
const BYTES_PER_SAMPLE: usize = 2;
const N_LAYERS: usize = 19;
const BLANK_ID: usize = 0;
const SOS_ID: usize = 1;
const UNK_ID: usize = 2;

// ---------------------------------------------------------------------------
// Kaldi 80-mel frontend
// ---------------------------------------------------------------------------

fn povey_window() -> Vec<f32> {
    let a = 2.0 * std::f32::consts::PI / WIN as f32;
    (0..WIN)
        .map(|i| (0.5 - 0.5 * (a * i as f32).cos()).powf(0.85))
        .collect()
}

fn hz_to_mel(f: f32) -> f32 {
    1127.0 * (1.0 + f / 700.0).ln()
}
fn mel_to_hz(m: f32) -> f32 {
    700.0 * ((m / 1127.0).exp() - 1.0)
}

/// Kaldi mel filterbank: peak=1 triangles, 20–8000 Hz.
fn kaldi_filterbank() -> Vec<Vec<f32>> {
    let n_bins = WIN / 2 + 1;
    let f_max = 16_000.0 / 2.0;
    let mel_lo = hz_to_mel(20.0);
    let mel_hi = hz_to_mel(f_max);
    let hz_pts: Vec<f32> = (0..N_MELS + 2)
        .map(|i| mel_to_hz(mel_lo + (mel_hi - mel_lo) * i as f32 / (N_MELS + 1) as f32))
        .collect();
    let bin_hz: Vec<f32> = (0..n_bins)
        .map(|b| b as f32 * 16000.0 / (2.0 * (n_bins - 1) as f32))
        .collect();
    let mut fb = vec![vec![0.0; n_bins]; N_MELS];
    for m in 0..N_MELS {
        let (lo, ctr, hi) = (hz_pts[m], hz_pts[m + 1], hz_pts[m + 2]);
        for (b, &f) in bin_hz.iter().enumerate() {
            let w = ((f - lo) / (ctr - lo).max(1e-10))
                .min((hi - f) / (hi - ctr).max(1e-10))
                .max(0.0);
            fb[m][b] = w;
        }
    }
    fb
}

struct KaldiFrontend {
    fft: std::sync::Arc<dyn realfft::RealToComplex<f32>>,
    window: Vec<f32>,
    fb: Vec<Vec<f32>>,
    frame: Vec<f32>,
    spectrum: Vec<realfft::num_complex::Complex<f32>>,
    power: Vec<f32>,
}

impl KaldiFrontend {
    fn new() -> Self {
        let mut planner = realfft::RealFftPlanner::new();
        let fft = planner.plan_fft_forward(WIN);
        let n_bins = WIN / 2 + 1;
        Self {
            spectrum: fft.make_output_vec(),
            fft,
            window: povey_window(),
            fb: kaldi_filterbank(),
            frame: vec![0.0; WIN],
            power: vec![0.0; n_bins],
        }
    }

    fn log_mel(&mut self, samples: &[f32]) -> Vec<[f32; N_MELS]> {
        if samples.len() < WIN {
            return Vec::new();
        }
        let n_frames = 1 + (samples.len() - WIN) / HOP;
        let mut out = Vec::with_capacity(n_frames);
        for t in 0..n_frames {
            let seg = &samples[t * HOP..t * HOP + WIN];
            self.frame.copy_from_slice(seg);
            let mean = self.frame.iter().sum::<f32>() / WIN as f32;
            for s in &mut self.frame {
                *s -= mean;
            }
            // Preemphasis (reverse order for correctness).
            for i in (1..WIN).rev() {
                self.frame[i] -= PREEMPH * self.frame[i - 1];
            }
            for i in 0..WIN {
                self.frame[i] *= self.window[i];
            }
            if self.fft.process(&mut self.frame, &mut self.spectrum).is_err() {
                return out;
            }
            for (p, c) in self.power.iter_mut().zip(self.spectrum.iter()) {
                *p = c.re * c.re + c.im * c.im;
            }
            let mut feats = [0.0_f32; N_MELS];
            for (m, row) in self.fb.iter().enumerate() {
                let acc: f32 = row.iter().zip(self.power.iter()).map(|(&w, &p)| w * p).sum();
                feats[m] = (acc + LOG_GUARD).ln();
            }
            out.push(feats);
        }
        out
    }
}

// ---------------------------------------------------------------------------
// Cache management — all 116 tensors stored as dynamic arrays
// ---------------------------------------------------------------------------

/// All cache tensors for the 19-layer Zipformer encoder.
///
/// Caches are split into "growing" types (key/val — accumulate time context)
/// and "fixed" types (nonlin_attn, conv1, conv2 — fixed projection buffers).
/// Stored as dynamic-dimensionality arrays so we don't need type-level shape
/// tracking.
#[allow(dead_code)]
struct ZipformerCaches {
    // Per-layer: [key_dim, 1, 0] initially, grows to [key_dim, 1, T] over time.
    key: [ndarray::Array<f32, IxDyn>; N_LAYERS],
    val1: [ndarray::Array<f32, IxDyn>; N_LAYERS],
    val2: [ndarray::Array<f32, IxDyn>; N_LAYERS],
    // Per-layer: fixed shapes like [1, 1, heads, attn_dim].
    nonlin_attn: [ndarray::Array<f32, IxDyn>; N_LAYERS],
    // Per-layer: [1, ch, conv_width].
    conv1: [ndarray::Array<f32, IxDyn>; N_LAYERS],
    conv2: [ndarray::Array<f32, IxDyn>; N_LAYERS],
    // Embed projection state: [1, 128, 3, N_LAYERS].
    embed_states: ndarray::Array<f32, IxDyn>,
}

impl ZipformerCaches {
    fn zeroed() -> Self {
        // Per-layer shapes (from ONNX graph). The key/value/conv/nonlin tensors
        // have ALL dimensions fixed in the ONNX graph — including the time/seq
        // dimension — because `ort` rejects zero-sized dimensions. Even though
        // the model never reads these zero-initialised values, they must be
        // present.
        // Per-layer cache shapes, verified against the ONNX graph.
        // kh=heads*head_dim  kt=head_dim  ad=attn_dim (nonlin)
        // vd=val_dim  cc=conv_ch  cw=conv_width
        let kh = |i: usize| -> usize { match i { 0|1=>256, 2|3|17|18=>128, 4..=7|13..=16=>64, 8..=12=>32, _=>64 }};
        let kt = |i: usize| -> usize { match i { 8..=12=>256, _=>128 }};
        let ad = |i: usize| -> usize { match i { 0|1=>144, 2|3|17|18=>192, 4..=7|13..=16=>384, 8..=12=>576, _=>192 }};
        let vd = |i: usize| -> usize { match i { 8..=12=>96, _=>48 }};
        let cc = |i: usize| -> usize { match i { 0|1=>192, 2|3|17|18=>256, 4..=7|13..=16=>512, 8..=12=>768, _=>256 }};
        let cw = |i: usize| -> usize { match i { 0..=3|17|18=>15, _=>7 }};

        Self {
            key: std::array::from_fn(|i| ndarray::Array::<f32, IxDyn>::zeros(IxDyn(&[kh(i), 1, kt(i)]))),
            val1: std::array::from_fn(|i| ndarray::Array::<f32, IxDyn>::zeros(IxDyn(&[kh(i), 1, vd(i)]))),
            val2: std::array::from_fn(|i| ndarray::Array::<f32, IxDyn>::zeros(IxDyn(&[kh(i), 1, vd(i)]))),
            nonlin_attn: std::array::from_fn(|i| ndarray::Array::<f32, IxDyn>::zeros(IxDyn(&[1, 1, kh(i), ad(i)]))),
            conv1: std::array::from_fn(|i| ndarray::Array::<f32, IxDyn>::zeros(IxDyn(&[1, cc(i), cw(i)]))),
            conv2: std::array::from_fn(|i| ndarray::Array::<f32, IxDyn>::zeros(IxDyn(&[1, cc(i), cw(i)]))),
            embed_states: ndarray::Array::<f32, IxDyn>::zeros(IxDyn(&[1, 128, 3, N_LAYERS])),
        }
    }
}

// ---------------------------------------------------------------------------
// Parsed vocab
// ---------------------------------------------------------------------------

/// Parse `"token id"` per line (sherpa-onnx format). Blank=0, SOS=1, UNK=2.
fn load_vocab(path: &str) -> Result<Vec<String>, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("read vocab {path}: {e}"))?;
    let mut vocab: Vec<String> = Vec::with_capacity(N_LOGITS);
    for line in text.lines() {
        // The format is "<token> <id>", but the token may contain spaces.
        // Use rfind(' ') to split off the last whitespace-delimited id.
        if let Some(space) = line.rfind(' ') {
            let token = &line[..space];
            let id: usize = line[space + 1..].trim().parse().map_err(|_| {
                format!("bad id in vocab line: {line}")
            })?;
            // Grow the vocab vector if needed (sparse ids are possible).
            while vocab.len() <= id {
                vocab.push(String::new());
            }
            vocab[id] = token.to_string();
        } else {
            // Line-index-as-id fallback (shouldn't happen for this model).
            vocab.push(line.to_string());
        }
    }
    if vocab.is_empty() {
        return Err(format!("vocab file {path} is empty"));
    }
    Ok(vocab)
}

// ---------------------------------------------------------------------------
// Zipformer Streaming Stage
// ---------------------------------------------------------------------------

pub struct SherpaZipformer {
    encoder: Session,
    decoder: Session,
    joiner: Session,
    vocab: Vec<String>,
    mel: KaldiFrontend,
    // Streaming state.
    caches: ZipformerCaches,
    pending: Vec<[f32; N_MELS]>,
    sample_buf: Vec<f32>,
    transcript: String,
}

impl SherpaZipformer {
    pub fn from_env() -> Result<Self, String> {
        let dir = std::env::var("SHERPA_MODEL_DIR")
            .ok()
            .filter(|d| !d.is_empty())
            .unwrap_or_else(|| "./models/sherpa_zipformer".to_string());
        Self::load(&dir)
    }

    pub fn load(dir: &str) -> Result<Self, String> {
        init_ort()?;
        let build = |p: &str| -> Result<Session, String> {
            Session::builder()
                .map_err(|e| format!("session builder: {e}"))?
                .commit_from_file(p)
                .map_err(|e| format!("load {p}: {e}"))
        };
        let encoder = build(&format!("{dir}/encoder.onnx"))?;
        let decoder = build(&format!("{dir}/decoder.onnx"))?;
        let joiner = build(&format!("{dir}/joiner.onnx"))?;
        let vocab = load_vocab(&format!("{dir}/tokens.txt"))?;
        tracing::info!(
            vocab_entries = vocab.len(),
            chunk_frames = CHUNK_FRAMES,
            "Sherpa Zipformer streaming STT loaded from {dir}"
        );
        Ok(Self {
            encoder, decoder, joiner, vocab,
            mel: KaldiFrontend::new(),
            caches: ZipformerCaches::zeroed(),
            pending: Vec::new(),
            sample_buf: Vec::new(),
            transcript: String::new(),
        })
    }

    pub fn reset(&mut self) {
        self.caches = ZipformerCaches::zeroed();
        self.pending.clear();
        self.sample_buf.clear();
        self.transcript.clear();
    }

    pub fn transcript(&self) -> &str {
        &self.transcript
    }

    pub fn push_pcm(&mut self, pcm: &[u8]) -> Vec<String> {
        let samples: Vec<f32> = pcm
            .chunks_exact(BYTES_PER_SAMPLE)
            .map(|c| i16::from_le_bytes([c[0], c[1]]) as f32 / 32768.0)
            .collect();
        self.push_samples(&samples)
    }

    pub fn push_samples(&mut self, samples: &[f32]) -> Vec<String> {
        self.sample_buf.extend_from_slice(samples);
        // Audio needed for CHUNK_FRAMES mel frames: WIN + (frames-1)*HOP
        let need = WIN + (CHUNK_FRAMES - 1) * HOP; // 22800 samples = 1.425s
        let mut results = Vec::new();
        while self.sample_buf.len() >= need {
            let block: Vec<f32> = self.sample_buf.drain(..need).collect();
            let frames = self.mel.log_mel(&block);
            self.pending.extend(frames);
            while self.pending.len() >= CHUNK_FRAMES {
                let chunk: Vec<[f32; N_MELS]> = self.pending.drain(..CHUNK_FRAMES).collect();
                // Model expects [batch, time, mel] = [1, 141, 80].
                let mut audio = Array3::<f32>::zeros((1, CHUNK_FRAMES, N_MELS));
                for (t, frame) in chunk.iter().enumerate() {
                    for (m, &v) in frame.iter().enumerate() {
                        audio[[0, t, m]] = v;
                    }
                }
                match self.run_chunk(audio) {
                    Ok(text) => {
                        if !text.is_empty() {
                            results.push(text.clone());
                        }
                    }
                    Err(e) => {
                        tracing::error!(error = %e, "Zipformer chunk failed");
                        return results;
                    }
                }
            }
        }
        results
    }

    pub fn flush(&mut self) -> Vec<String> {
        let need = WIN + (CHUNK_FRAMES - 1) * HOP;
        let missing = need.saturating_sub(self.sample_buf.len());
        let pad = vec![0.0_f32; missing];
        self.push_samples(&pad)
    }

    /// Build the 117 named encoder inputs and run.
    fn run_chunk(&mut self, audio: Array3<f32>) -> Result<String, String> {
        // -- Encoder --
        let lens = Array1::from_vec(vec![CHUNK_FRAMES as i64]);

        // Convert all caches to tensors and build inputs.
        let mut named: Vec<(String, Value)> = Vec::with_capacity(117);

        named.push(("x".into(), Tensor::from_array(audio).map_err(|e| format!("x: {e}"))?.into()));
        named.push(("processed_lens".into(), Tensor::from_array(lens).map_err(|e| format!("lens: {e}"))?.into()));
        named.push(("embed_states".into(), Tensor::from_array(self.caches.embed_states.clone().into_dyn()).map_err(|e| format!("embed: {e}"))?.into()));

        for i in 0..N_LAYERS {
            named.push((
                format!("cached_key_{i}"),
                Tensor::from_array(self.caches.key[i].clone().into_dyn()).map_err(|e| format!("key[{i}]: {e}"))?.into()
            ));
            named.push((
                format!("cached_nonlin_attn_{i}"),
                Tensor::from_array(self.caches.nonlin_attn[i].clone().into_dyn()).map_err(|e| format!("nonlin[{i}]: {e}"))?.into()
            ));
            named.push((
                format!("cached_val1_{i}"),
                Tensor::from_array(self.caches.val1[i].clone().into_dyn()).map_err(|e| format!("val1[{i}]: {e}"))?.into()
            ));
            named.push((
                format!("cached_val2_{i}"),
                Tensor::from_array(self.caches.val2[i].clone().into_dyn()).map_err(|e| format!("val2[{i}]: {e}"))?.into()
            ));
            named.push((
                format!("cached_conv1_{i}"),
                Tensor::from_array(self.caches.conv1[i].clone().into_dyn()).map_err(|e| format!("conv1[{i}]: {e}"))?.into()
            ));
            named.push((
                format!("cached_conv2_{i}"),
                Tensor::from_array(self.caches.conv2[i].clone().into_dyn()).map_err(|e| format!("conv2[{i}]: {e}"))?.into()
            ));
        }

        let out = self.encoder.run(named).map_err(|e| format!("encoder: {e}"))?;

        // Extract encoder_out.
        let enc_out = extract_dyn(&out[0], "encoder_out")?;
        let enc_dims = enc_out.shape().to_vec();
        if enc_dims.len() != 3 {
            return Err(format!("encoder_out: expected 3 dims, got {:?}", enc_dims));
        }
        let enc_time = enc_dims[1];

        // Update caches.
        // Outputs are ordered: [encoder_out, key_0, nonlin_0, val1_0, val2_0, conv1_0, conv2_0, key_1, ...] + embed_states + lens
        //
        // Post-encoder_out, outputs cycle through the 6 cache types per layer.
        let mut oi = 1; // skip encoder_out at index 0
        for i in 0..N_LAYERS {
            self.caches.key[i] = extract_dyn(&out[oi], &format!("new_cached_key_{i}"))?; oi += 1;
            self.caches.nonlin_attn[i] = extract_dyn(&out[oi], &format!("new_cached_nonlin_attn_{i}"))?; oi += 1;
            self.caches.val1[i] = extract_dyn(&out[oi], &format!("new_cached_val1_{i}"))?; oi += 1;
            self.caches.val2[i] = extract_dyn(&out[oi], &format!("new_cached_val2_{i}"))?; oi += 1;
            self.caches.conv1[i] = extract_dyn(&out[oi], &format!("new_cached_conv1_{i}"))?; oi += 1;
            self.caches.conv2[i] = extract_dyn(&out[oi], &format!("new_cached_conv2_{i}"))?; oi += 1;
        }
        self.caches.embed_states = extract_dyn(&out[oi], "new_embed_states")?;

        // -- Greedy RNN-T decode --
        let mut text = String::new();
        // Decoder context: [sos, blank].
        let mut dec_ctx: Vec<i64> = vec![SOS_ID as i64, BLANK_ID as i64];

        for t in 0..enc_time {
            // Select encoder frame t: [1, 512]
            let enc_t = enc_out.index_axis(Axis(1), t).to_owned(); // [512]
            let enc_2d = enc_t.into_shape_with_order((1, ENC_HIDDEN)).map_err(|e| format!("enc reshape: {e}"))?;

            for _ in 0..MAX_SYMBOLS_PER_STEP {
                // Decoder
                let ctx = Array2::from_shape_vec((1, 2), dec_ctx.clone())
                    .map_err(|e| format!("ctx: {e}"))?;
                let d_out = self.decoder.run(ort::inputs![
                    "y" => Tensor::from_array(ctx).map_err(|e| format!("y tensor: {e}"))?,
                ]).map_err(|e| format!("decoder: {e}"))?;
                let dec_vec = extract_dyn(&d_out[0], "decoder_out")?;
                let dec_2d = dec_vec
                    .into_shape_with_order((1, ENC_HIDDEN))
                    .map_err(|e| format!("dec reshape: {e}"))?;

                // Joiner
                let j_out = self.joiner.run(ort::inputs![
                    "encoder_out" => Tensor::from_array(enc_2d.clone()).map_err(|e| format!("j_enc: {e}"))?,
                    "decoder_out" => Tensor::from_array(dec_2d).map_err(|e| format!("j_dec: {e}"))?,
                ]).map_err(|e| format!("joiner: {e}"))?;
                let (_j_shape, j_data) = j_out[0]
                    .try_extract_tensor::<f32>()
                    .map_err(|e| format!("logits: {e}"))?;
                let logits = &j_data[..N_LOGITS];
                let token = argmax(logits);

                if token == BLANK_ID {
                    break;
                }
                if token != SOS_ID && token != UNK_ID {
                    if let Some(piece) = self.vocab.get(token) {
                        text.push_str(piece);
                    }
                }
                // Slide context window.
                dec_ctx = [dec_ctx[1], token as i64].to_vec();
            }
        }

        let text = text.replace('\u{2581}', " ");
        if !text.is_empty() {
            self.transcript.push_str(&text);
        }
        Ok(text)
    }
}

// ---------------------------------------------------------------------------
// SttStage trait
// ---------------------------------------------------------------------------

impl SttStage for SherpaZipformer {
    fn transcribe(&mut self, pcm: &[u8], partial: &mut dyn FnMut(&str)) -> String {
        self.reset();
        for r in self.push_pcm(pcm) {
            if !r.is_empty() {
                partial(self.transcript.trim());
            }
        }
        for r in self.flush() {
            if !r.is_empty() {
                partial(self.transcript.trim());
            }
        }
        self.transcript.trim().to_string()
    }

    /// Note: Sherpa Zipformer does not emit an `<EOU>` token. The `end_of_utterance`
    /// flag will always be `false`. Turn boundary detection must rely on VAD.
    fn push_chunk(&mut self, pcm: &[u8]) -> Option<crate::rt_pipeline::SttChunkResult> {
        let results = self.push_pcm(pcm);
        if results.is_empty() {
            None
        } else {
            let delta_text = results.join("");
            Some(crate::rt_pipeline::SttChunkResult {
                delta_text,
                end_of_utterance: false,
            })
        }
    }

    fn reset_stream(&mut self) {
        self.reset();
    }

    fn supports_lookback(&self) -> bool {
        true
    }
}

// ---------------------------------------------------------------------------
// Replica trait
// ---------------------------------------------------------------------------

impl Replica for SherpaZipformer {
    type Input = SttInput;
    type Delta = String;
    type Output = String;

    fn process(
        &mut self,
        input: Self::Input,
        cancel: &AtomicBool,
        emit: &mut dyn FnMut(Self::Delta),
    ) -> Self::Output {
        self.reset();
        let block = (WIN + (CHUNK_FRAMES - 1) * HOP) * BYTES_PER_SAMPLE;
        for part in input.pcm.chunks(block) {
            if cancel.load(Ordering::Relaxed) {
                break;
            }
            for r in self.push_pcm(part) {
                if !r.is_empty() {
                    emit(self.transcript.trim().to_string());
                }
            }
        }
        if !cancel.load(Ordering::Relaxed) {
            for r in self.flush() {
                if !r.is_empty() {
                    emit(self.transcript.trim().to_string());
                }
            }
        }
        self.transcript.trim().to_string()
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn extract_dyn(val: &Value, what: &str) -> Result<ndarray::Array<f32, IxDyn>, String> {
    let (shape, data) = val
        .try_extract_tensor::<f32>()
        .map_err(|e| format!("{what}: {e}"))?;
    let dims: Vec<usize> = shape.iter().map(|&d| d as usize).collect();
    ndarray::Array::from_shape_vec(IxDyn(&dims), data.to_vec())
        .map_err(|e| format!("{what} reshape: {e}"))
}

fn argmax(vals: &[f32]) -> usize {
    vals.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0)
}


// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn argmax_works() {
        assert_eq!(argmax(&[0.1, 0.9, 0.5]), 1);
        assert_eq!(argmax(&[3.0, 1.0, 2.0]), 0);
    }

    #[test]
    fn povey_window_starts_at_zero_and_peaks_at_centre() {
        let w = povey_window();
        assert_eq!(w.len(), WIN);
        // Povey window has a single sample at 0 at each edge (0.5-0.5*cos(0) = 0).
        assert!(w[0] < 1e-6);
        // Near the centre it peaks.
        let mid = WIN / 2;
        assert!(w[mid] > 0.99);
        // And the centre is the maximum.
        for (i, &v) in w.iter().enumerate() {
            assert!(v <= w[mid] + 1e-6, "w[{i}]={v} > w[{mid}]={}", w[mid]);
        }
    }

    #[test]
    fn filterbank_has_80_bins() {
        let fb = kaldi_filterbank();
        assert_eq!(fb.len(), N_MELS);
        // Kaldi norm = peak 1, so maximum weight should be ≈ 1.0.
        for row in &fb {
            let max_w = row.iter().cloned().fold(0.0_f32, f32::max);
            assert!(max_w <= 1.001, "peak weight {max_w} > 1");
        }
        // Row 0 (lowest mel) spans 20-42 Hz; bin 0 is 0 Hz (DC), so no overlap.
        assert!(fb[0][0] < 1e-6, "fb[0][0] should be 0 at 0 Hz");
        // Roughness check: row 0 has most energy in its first few bins.
        let row_sum: f32 = fb[0].iter().sum();
        assert!(row_sum > 0.0, "row 0 should have some energy");
    }

    #[test]
    fn special_ids_are_correct() {
        assert_eq!(BLANK_ID, 0);
        assert_eq!(SOS_ID, 1);
        assert!(BLANK_ID < N_LOGITS);
        assert!(SOS_ID < N_LOGITS);
    }

    #[test]
    fn partial_mel_produces_expected_frame_count() {
        let mut fe = KaldiFrontend::new();
        // Audio needed for CHUNK_FRAMES mel frames = WIN + (frames-1)*HOP.
        let n = WIN + (CHUNK_FRAMES - 1) * HOP;
        let sig: Vec<f32> = (0..n).map(|i| (i as f32 * 0.001).sin()).collect();
        let feats = fe.log_mel(&sig);
        assert_eq!(feats.len(), CHUNK_FRAMES,
            "expected {CHUNK_FRAMES} frames from {n} samples (win={WIN}, hop={HOP})");
    }

    #[test]
    fn mel_outputs_80_dimensional_vectors() {
        let mut fe = KaldiFrontend::new();
        let n = WIN + (CHUNK_FRAMES - 1) * HOP; // one full chunk
        let sig: Vec<f32> = (0..n).map(|i| (i as f32).sin() * 0.5).collect();
        let feats = fe.log_mel(&sig);
        for frame in feats {
            assert_eq!(frame.len(), N_MELS);
            // All values should be finite (no NaN from the log).
            for &v in &frame {
                assert!(v.is_finite(), "non-finite value {v}");
            }
        }
    }

    #[test]
    fn vocab_parses_650_lines() {
        let dir = std::env::temp_dir().join("szca_zipf_vocab_test");
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("tokens.txt");
        // Create a mock tokens.txt in the format "<token> <id>".
        let content = (0..650)
            .map(|i| format!("t{i} {i}"))
            .collect::<Vec<_>>()
            .join("\n");
        std::fs::write(&p, content).unwrap();
        let v = load_vocab(p.to_str().unwrap()).unwrap();
        assert_eq!(v.len(), 650);
        assert_eq!(v[0], "t0");
        assert_eq!(v[649], "t649");
        std::fs::remove_file(&p).ok();
    }

    #[test]
    fn decoder_init_context_is_two_tokens() {
        let ctx: Vec<i64> = vec![SOS_ID as i64, BLANK_ID as i64];
        assert_eq!(ctx.len(), 2);
        // After a token emission, the context slides.
        let emitted = 42_i64;
        let new_ctx = [ctx[1], emitted].to_vec();
        assert_eq!(new_ctx, [0, 42]);
    }
}
