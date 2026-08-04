/// Cache-aware streaming STT: Parakeet EOU 120M (FP16 encoder + INT8 decoder_joint).
///
/// This is the STREAMING counterpart to `rt_stt.rs`. The difference is structural,
/// not cosmetic:
///
/// | | `rt_stt.rs` (Parakeet TDT 0.6B) | this file (Parakeet EOU 120M) |
/// |---|---|---|
/// | Encoder context | full-sequence attention | **cache-aware**, 70-frame left context |
/// | Input granularity | the whole utterance, post-VAD | **1.28 s chunks, incrementally** |
/// | Turn end | server VAD silence timeout | **`<EOU>` token from the model itself** |
/// | Frontend | `nemo128.onnx` (per-utterance norm) | `rt_stt_mel` (raw log-mel) |
///
/// Because the encoder carries its own attention/conv cache between calls, audio
/// can be fed as it arrives — the encoder never re-reads earlier audio, so cost
/// is linear in stream length instead of quadratic in utterance length.
///
/// ```text
/// PCM16 mono 16 kHz (streamed)
///   → rt_stt_mel::MelFrontend       (128-bin raw log-mel, NO normalization)
///   → ring buffer, 128 mel frames   (= 1.28 s hop)
///   → encoder.onnx + 3 caches       (FP16, [1,512,T'] out; caches fed back)
///   → decoder_joint.onnx            (RNN-T greedy, per encoder frame)
///   → vocab.txt (line index = id)   ('▁' → space)
/// ```
///
/// ## Why the FP16 encoder and not the smaller INT8 one
///
/// The commonly linked INT8 export is built from `ConvInteger`/`MatMulInteger`.
/// ONNX Runtime has no CPU kernel for signed-INT8 `ConvInteger` before **1.24**,
/// and we run ORT **1.22** (`ort` 2.0.0-rc.10 → `ORT_API_VERSION` 22). Measured:
/// 1.19/1.22/1.23 fail at session creation, 1.24+ works — on arm64 **and**
/// x86_64. The FP16 export has no integer ops and runs at ~21× realtime on an
/// M-series CPU, comfortably inside the 1.28 s chunk budget. See PROJECT.md §16.
///
/// ## Two traps encoded here as constants
///
/// 1. **`outputs` slot indexing.** `decoder_joint` returns
///    `[1, 1, target_plus_sos, 1027]`. Slot 0 is the SOS position; the **last**
///    slot is the prediction for the supplied target. Reading slot 0 decodes
///    `"he wor worww"` instead of `"hello world"` — wrong, but plausible enough
///    to ship.
/// 2. **INT32, not INT64.** `audio_length` and `cache_last_channel_len` are
///    INT32 in this export (the INT8 export used INT64). ORT rejects the call
///    outright on a width mismatch, which surfaces as an empty transcript.
use std::sync::atomic::{AtomicBool, Ordering};

use ndarray::{Array1, Array3, Array4};
use ort::session::Session;
use ort::value::Tensor;

use crate::onnx::init_ort;
use crate::rt_pipeline::SttStage;
use crate::rt_stt::SttInput;
use crate::rt_stt_mel::{MelFrontend, N_MELS};
use crate::stage_pool::Replica;

/// Mel frames per encoder call (1.28 s at a 10 ms hop).
const CHUNK_FRAMES: usize = 128;

/// Width of the mel pre-encode cache carried between chunks.
const PRE_CACHE: usize = 16;

/// Encoder layer count (first dim of both attention caches).
const ENC_LAYERS: usize = 17;

/// Left-context frames retained in the channel cache.
const CACHE_CHANNEL: usize = 70;

/// Encoder hidden width.
const ENC_HIDDEN: usize = 512;

/// Conv cache depth.
const CACHE_TIME: usize = 8;

/// Prediction-network hidden width.
const PRED_HIDDEN: usize = 640;

/// Total logits: 1024 BPE + `<EOU>` + `<EOB>` + blank.
const N_LOGITS: usize = 1027;

/// Blank token id (RNN-T "advance time" symbol).
const BLANK_ID: usize = 1026;

/// End-of-utterance token — the reason this model exists.
const EOU_ID: usize = 1024;

/// End-of-boundary token (segment break inside a turn).
const EOB_ID: usize = 1025;

/// Safety cap on tokens emitted per encoder frame.
const MAX_SYMBOLS_PER_STEP: usize = 10;

/// Bytes per PCM16 sample.
const BYTES_PER_SAMPLE: usize = 2;

/// The three encoder caches plus the mel pre-encode cache. Feeding the outputs
/// of one call straight back as the inputs of the next IS the streaming
/// mechanism — drop them and every chunk is transcribed as if it were the start
/// of a fresh utterance.
struct EncoderCache {
    pre: Array3<f32>,          // [1, 128, 16]
    last_channel: Array4<f32>, // [17, 1, 70, 512]
    last_time: Array4<f32>,    // [17, 1, 512, 8]
    len: Array1<i32>,          // [1]  — INT32, not INT64
}

impl EncoderCache {
    fn zeroed() -> Self {
        Self {
            pre: Array3::zeros((1, N_MELS, PRE_CACHE)),
            last_channel: Array4::zeros((ENC_LAYERS, 1, CACHE_CHANNEL, ENC_HIDDEN)),
            last_time: Array4::zeros((ENC_LAYERS, 1, ENC_HIDDEN, CACHE_TIME)),
            len: Array1::zeros(1),
        }
    }
}

/// Decoder (prediction network) LSTM state plus the last emitted token.
struct DecoderState {
    s1: Array3<f32>, // [1, 1, 640]
    s2: Array3<f32>, // [1, 1, 640]
    last_token: i32,
}

impl DecoderState {
    fn primed() -> Self {
        Self {
            s1: Array3::zeros((1, 1, PRED_HIDDEN)),
            s2: Array3::zeros((1, 1, PRED_HIDDEN)),
            last_token: BLANK_ID as i32,
        }
    }
}

/// What a streaming chunk produced.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct EouChunkResult {
    /// Newly decoded text for this chunk (may be empty).
    pub text: String,
    /// The model signalled end-of-utterance — the turn can be closed WITHOUT
    /// waiting for a VAD silence timeout.
    pub end_of_utterance: bool,
}

/// Loaded streaming EOU engine: two ONNX sessions + vocab + a live stream state.
pub struct ParakeetEouStt {
    encoder: Session,
    decoder_joint: Session,
    vocab: Vec<String>,
    mel: MelFrontend,

    // ---- live streaming state ----
    cache: EncoderCache,
    dec: DecoderState,
    /// Mel frames not yet consumed by a full CHUNK_FRAMES encoder call.
    pending: Vec<[f32; N_MELS]>,
    /// PCM samples left over from a partial mel hop.
    sample_carry: Vec<f32>,
    /// Pre-allocated scratch buffer to avoid per-frame allocation in push_pcm.
    scratch: Vec<f32>,
    /// Transcript accumulated across the current utterance.
    transcript: String,
}

impl ParakeetEouStt {
    /// Load from env:
    ///   * `STT_EOU_MODEL_DIR` — model directory (default `./models/stt_eou`)
    ///
    /// Expected files: `encoder.onnx`, `decoder_joint.onnx`, `vocab.txt`.
    pub fn from_env() -> Result<Self, String> {
        let dir = std::env::var("STT_EOU_MODEL_DIR")
            .ok()
            .filter(|d| !d.is_empty())
            .unwrap_or_else(|| "./models/stt_eou".to_string());
        Self::load(&dir)
    }

    /// Load the two graphs and the vocab from `dir`.
    pub fn load(dir: &str) -> Result<Self, String> {
        init_ort()?;

        let build = |p: &str| -> Result<Session, String> {
            Session::builder()
                .map_err(|e| format!("session builder: {e}"))?
                .commit_from_file(p)
                .map_err(|e| format!("load {p}: {e}"))
        };

        let encoder = build(&format!("{dir}/encoder.onnx"))?;
        let decoder_joint = build(&format!("{dir}/decoder_joint.onnx"))?;
        let vocab = load_vocab_lines(&format!("{dir}/vocab.txt"))?;

        tracing::info!(
            vocab_entries = vocab.len(),
            chunk_ms = CHUNK_FRAMES * 10,
            "Parakeet EOU streaming STT loaded from {dir}"
        );

        Ok(Self {
            encoder,
            decoder_joint,
            vocab,
            mel: MelFrontend::new(),
            cache: EncoderCache::zeroed(),
            dec: DecoderState::primed(),
            pending: Vec::new(),
            sample_carry: Vec::new(),
            scratch: Vec::with_capacity(1024),
            transcript: String::new(),
        })
    }

    /// Drop all streaming state so the next `push_pcm` starts a fresh utterance.
    ///
    /// MUST be called between turns. Leaving the encoder cache populated leaks
    /// the previous turn's acoustic context into the next one.
    pub fn reset(&mut self) {
        self.cache = EncoderCache::zeroed();
        self.dec = DecoderState::primed();
        self.pending.clear();
        self.sample_carry.clear();
        self.scratch.clear();
        self.transcript.clear();
    }

    /// The transcript accumulated so far in this utterance.
    pub fn transcript(&self) -> &str {
        &self.transcript
    }

    /// Feed PCM16 bytes; run the encoder for every whole chunk that is now
    /// available. Returns one result per completed chunk.
    ///
    /// Audio shorter than a chunk is buffered, so callers can push arbitrarily
    /// small frames (e.g. 20 ms WebSocket writes) without special-casing.
    pub fn push_pcm(&mut self, pcm: &[u8]) -> Vec<EouChunkResult> {
        // Pre-allocate scratch for reuse across calls to reduce allocation churn.
        // We drain it before push_samples to avoid borrow conflict with &mut self.
        let n = pcm.len() / BYTES_PER_SAMPLE;
        self.scratch.clear();
        self.scratch.reserve(n);
        for chunk in pcm.chunks_exact(BYTES_PER_SAMPLE) {
            self.scratch.push(i16::from_le_bytes([chunk[0], chunk[1]]) as f32 / 32768.0);
        }
        let samples = std::mem::take(&mut self.scratch); // move out to break borrow
        let result = self.push_samples(&samples);
        self.scratch = samples; // put back for reuse (keeps allocation)
        self.scratch.clear();
        result
    }

    /// Same as [`Self::push_pcm`] but for f32 samples already in [-1, 1].
    pub fn push_samples(&mut self, samples: &[f32]) -> Vec<EouChunkResult> {
        // The mel frontend reflect-pads whatever it is handed, so feeding it
        // successive slices would inject an artificial edge at every boundary.
        // Accumulate and only extract on chunk-sized blocks.
        self.sample_carry.extend_from_slice(samples);

        let mut results = Vec::new();
        // Samples needed for one full encoder chunk at a 10 ms hop.
        let hop = crate::rt_stt_mel::HOP;
        let need = CHUNK_FRAMES * hop;

        while self.sample_carry.len() >= need {
            let block: Vec<f32> = self.sample_carry.drain(..need).collect();
            let frames = self.mel.log_mel(&block);
            self.pending.extend(frames);

            while self.pending.len() >= CHUNK_FRAMES {
                let chunk: Vec<[f32; N_MELS]> = self.pending.drain(..CHUNK_FRAMES).collect();
                match self.run_chunk(&chunk) {
                    Ok(r) => results.push(r),
                    Err(e) => {
                        tracing::error!(error = %e, "EOU streaming chunk failed");
                        return results;
                    }
                }
            }
        }
        results
    }

    /// Flush buffered audio by zero-padding to a full chunk, then reset.
    ///
    /// Used at end-of-turn so a trailing partial chunk still gets transcribed.
    pub fn flush(&mut self) -> Vec<EouChunkResult> {
        let hop = crate::rt_stt_mel::HOP;
        if self.sample_carry.is_empty() && self.pending.is_empty() {
            return Vec::new();
        }
        let need = CHUNK_FRAMES * hop;
        let missing = need.saturating_sub(self.sample_carry.len());
        let pad = vec![0.0_f32; missing];
        self.push_samples(&pad)
    }

    /// Run one encoder chunk + the RNN-T greedy decode over its output frames.
    fn run_chunk(&mut self, chunk: &[[f32; N_MELS]]) -> Result<EouChunkResult, String> {
        // Encoder wants [1, 128, 128] = [batch, mel, time]; `chunk` is time-major.
        let mut audio = Array3::<f32>::zeros((1, N_MELS, CHUNK_FRAMES));
        for (t, frame) in chunk.iter().enumerate() {
            for (m, &v) in frame.iter().enumerate() {
                audio[[0, m, t]] = v;
            }
        }

        let (enc_out, n_frames) = {
            let out = self
                .encoder
                .run(ort::inputs![
                    "audio_signal" => Tensor::from_array(audio).map_err(|e| format!("audio tensor: {e}"))?,
                    "audio_length" => Tensor::from_array(Array1::from_vec(vec![CHUNK_FRAMES as i32])).map_err(|e| format!("audio_length tensor: {e}"))?,
                    "pre_cache" => Tensor::from_array(self.cache.pre.clone()).map_err(|e| format!("pre_cache tensor: {e}"))?,
                    "cache_last_channel" => Tensor::from_array(self.cache.last_channel.clone()).map_err(|e| format!("cache_channel tensor: {e}"))?,
                    "cache_last_time" => Tensor::from_array(self.cache.last_time.clone()).map_err(|e| format!("cache_time tensor: {e}"))?,
                    "cache_last_channel_len" => Tensor::from_array(self.cache.len.clone()).map_err(|e| format!("cache_len tensor: {e}"))?,
                ])
                .map_err(|e| format!("encoder forward: {e}"))?;

            // encoded_output is [1, 512, T'].
            let (shape, data) = out[0]
                .try_extract_tensor::<f32>()
                .map_err(|e| format!("encoded_output: {e}"))?;
            let dims: Vec<usize> = shape.iter().map(|&d| d as usize).collect();
            let enc = Array3::from_shape_vec((dims[0], dims[1], dims[2]), data.to_vec())
                .map_err(|e| format!("encoded_output reshape: {e}"))?;

            let (_, len_data) = out[1]
                .try_extract_tensor::<i32>()
                .map_err(|e| format!("encoded_length: {e}"))?;
            let n = *len_data.first().ok_or("encoded_length empty")? as usize;

            // Carry the caches forward — this is what makes it streaming.
            self.cache.pre = extract3(&out[2], "new_pre_cache")?;
            self.cache.last_channel = extract4(&out[3], "new_cache_last_channel")?;
            self.cache.last_time = extract4(&out[4], "new_cache_last_time")?;
            let (_, nl) = out[5]
                .try_extract_tensor::<i32>()
                .map_err(|e| format!("new_cache_len: {e}"))?;
            self.cache.len = Array1::from_vec(nl.to_vec());

            (enc, n)
        };

        let valid = n_frames.min(enc_out.shape()[2]);
        let mut chunk_text = String::new();
        let mut eou = false;

        for t in 0..valid {
            // [1, 512, 1] — one encoder frame.
            let frame = enc_out
                .slice(ndarray::s![.., .., t..t + 1])
                .as_standard_layout()
                .into_owned();

            for _ in 0..MAX_SYMBOLS_PER_STEP {
                let token = self.decode_step(&frame)?;
                if token == BLANK_ID {
                    break; // advance time
                }
                match token {
                    EOU_ID => eou = true,
                    EOB_ID => {}
                    id => {
                        if let Some(piece) = self.vocab.get(id) {
                            chunk_text.push_str(piece);
                        }
                    }
                }
            }
        }

        let text = chunk_text.replace('\u{2581}', " ");
        if !text.is_empty() {
            self.transcript.push_str(&text);
        }
        Ok(EouChunkResult {
            text,
            end_of_utterance: eou,
        })
    }

    /// One decoder+joint step for the current target at one encoder frame.
    /// Updates the LSTM state and `last_token` when a real token is emitted.
    fn decode_step(&mut self, enc_frame: &Array3<f32>) -> Result<usize, String> {
        let targets = ndarray::Array2::from_shape_vec((1, 1), vec![self.dec.last_token])
            .map_err(|e| format!("targets shape: {e}"))?;

        let out = self
            .decoder_joint
            .run(ort::inputs![
                "encoder_outputs" => Tensor::from_array(enc_frame.clone()).map_err(|e| format!("enc_frame tensor: {e}"))?,
                "targets" => Tensor::from_array(targets).map_err(|e| format!("targets tensor: {e}"))?,
                "input_states_1" => Tensor::from_array(self.dec.s1.clone()).map_err(|e| format!("state1 tensor: {e}"))?,
                "input_states_2" => Tensor::from_array(self.dec.s2.clone()).map_err(|e| format!("state2 tensor: {e}"))?,
            ])
            .map_err(|e| format!("decoder_joint forward: {e}"))?;

        // outputs is [1, 1, target_plus_sos, 1027]. Slot 0 is the SOS position;
        // the LAST slot holds the prediction for our target. Using slot 0 here
        // decodes "he wor worww" instead of "hello world".
        let (shape, data) = out[0]
            .try_extract_tensor::<f32>()
            .map_err(|e| format!("joint logits: {e}"))?;
        let dims: Vec<usize> = shape.iter().map(|&d| d as usize).collect();
        let slots = *dims.get(2).unwrap_or(&1);
        if data.len() < N_LOGITS {
            return Err(format!("joint logits too small: {}", data.len()));
        }
        let start = (slots - 1) * N_LOGITS;
        let logits = &data[start..start + N_LOGITS];
        let token = argmax(logits);

        if token != BLANK_ID {
            self.dec.last_token = token as i32;
            self.dec.s1 = extract3(&out[1], "output_states_1")?;
            self.dec.s2 = extract3(&out[2], "output_states_2")?;
        }
        Ok(token)
    }
}

// ---------------------------------------------------------------------------
// SttStage — batch-compatible entry point
// ---------------------------------------------------------------------------

impl SttStage for ParakeetEouStt {
    /// Transcribe a complete utterance by streaming it through the chunked
    /// encoder. `partial` fires once per 1.28 s chunk with the growing
    /// transcript, so callers see genuine incremental results.
    fn transcribe(&mut self, pcm: &[u8], partial: &mut dyn FnMut(&str)) -> String {
        self.reset();
        for r in self.push_pcm(pcm) {
            if !r.text.is_empty() {
                partial(self.transcript.trim());
            }
        }
        for r in self.flush() {
            if !r.text.is_empty() {
                partial(self.transcript.trim());
            }
        }
        self.transcript.trim().to_string()
    }

    fn push_chunk(&mut self, pcm: &[u8]) -> Option<crate::rt_pipeline::SttChunkResult> {
        let results = self.push_pcm(pcm);
        if results.is_empty() {
            None
        } else {
            let mut delta_text = String::new();
            let mut end_of_utterance = false;
            for r in results {
                if !r.text.is_empty() {
                    delta_text.push_str(&r.text);
                }
                if r.end_of_utterance {
                    end_of_utterance = true;
                }
            }
            Some(crate::rt_pipeline::SttChunkResult {
                delta_text,
                end_of_utterance,
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
// StagePool Replica implementation
// ---------------------------------------------------------------------------

/// Deliberately the SAME input type as the full-utterance stage
/// (`rt_stt::SttInput`), so `SttBackend` can dispatch to either replica and the
/// pool, adapter and session code stay identical across backends.
impl Replica for ParakeetEouStt {
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

        let samples: Vec<f32> = input
            .pcm
            .chunks_exact(BYTES_PER_SAMPLE)
            .map(|c| i16::from_le_bytes([c[0], c[1]]) as f32 / 32768.0)
            .collect();

        // Feed in chunk-sized blocks so cancellation lands between encoder calls
        // rather than only at the end.
        let block = CHUNK_FRAMES * crate::rt_stt_mel::HOP;
        let mut done = false;
        for part in samples.chunks(block) {
            if cancel.load(Ordering::Relaxed) {
                done = true;
                break;
            }
            for r in self.push_samples(part) {
                if !r.text.is_empty() {
                    emit(self.transcript.trim().to_string());
                }
                if r.end_of_utterance {
                    done = true;
                }
            }
            if done {
                break;
            }
        }

        if !done && !cancel.load(Ordering::Relaxed) {
            for r in self.flush() {
                if !r.text.is_empty() {
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

/// Extract a 3-D f32 tensor from an ORT output.
fn extract3(
    val: &ort::value::DynValue,
    what: &str,
) -> Result<Array3<f32>, String> {
    let (shape, data) = val
        .try_extract_tensor::<f32>()
        .map_err(|e| format!("{what} extract: {e}"))?;
    let d: Vec<usize> = shape.iter().map(|&x| x as usize).collect();
    if d.len() != 3 {
        return Err(format!("{what}: expected 3 dims, got {}", d.len()));
    }
    Array3::from_shape_vec((d[0], d[1], d[2]), data.to_vec())
        .map_err(|e| format!("{what} reshape: {e}"))
}

/// Extract a 4-D f32 tensor from an ORT output.
fn extract4(
    val: &ort::value::DynValue,
    what: &str,
) -> Result<Array4<f32>, String> {
    let (shape, data) = val
        .try_extract_tensor::<f32>()
        .map_err(|e| format!("{what} extract: {e}"))?;
    let d: Vec<usize> = shape.iter().map(|&x| x as usize).collect();
    if d.len() != 4 {
        return Err(format!("{what}: expected 4 dims, got {}", d.len()));
    }
    Array4::from_shape_vec((d[0], d[1], d[2], d[3]), data.to_vec())
        .map_err(|e| format!("{what} reshape: {e}"))
}

/// Index of the largest value.
fn argmax(vals: &[f32]) -> usize {
    vals.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0)
}

/// Parse a one-piece-per-line vocab where the LINE INDEX is the token id.
///
/// This is NOT the format `models/stt/parakeet_vocab.txt` uses ("piece id"
/// pairs). Parsing this file with the pair parser yields an EMPTY vocab, which
/// then decodes as an empty transcript with no error anywhere — so the two
/// loaders stay deliberately separate.
fn load_vocab_lines(path: &str) -> Result<Vec<String>, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("read vocab {path}: {e}"))?;
    let vocab: Vec<String> = text.lines().map(|l| l.to_string()).collect();
    if vocab.is_empty() {
        return Err(format!("vocab file {path} is empty"));
    }
    Ok(vocab)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn argmax_picks_largest() {
        assert_eq!(argmax(&[0.1, 0.9, 0.5]), 1);
        assert_eq!(argmax(&[3.0, 1.0, 2.0]), 0);
        assert_eq!(argmax(&[-1.0, -2.0, -0.5]), 2);
    }

    #[test]
    fn cache_shapes_match_the_graph() {
        let c = EncoderCache::zeroed();
        assert_eq!(c.pre.dim(), (1, N_MELS, PRE_CACHE));
        assert_eq!(c.last_channel.dim(), (ENC_LAYERS, 1, CACHE_CHANNEL, ENC_HIDDEN));
        assert_eq!(c.last_time.dim(), (ENC_LAYERS, 1, ENC_HIDDEN, CACHE_TIME));
        assert_eq!(c.len.len(), 1);
    }

    #[test]
    fn decoder_state_primes_with_blank() {
        let d = DecoderState::primed();
        assert_eq!(d.last_token, BLANK_ID as i32);
        assert_eq!(d.s1.dim(), (1, 1, PRED_HIDDEN));
        assert_eq!(d.s2.dim(), (1, 1, PRED_HIDDEN));
    }

    /// These are all compile-time constants, so assert at compile time — a
    /// `const` block turns an out-of-range id into a build failure instead of a
    /// test failure.
    #[test]
    fn special_token_ids_are_distinct_and_in_range() {
        const _: () = {
            assert!(EOU_ID < N_LOGITS && EOB_ID < N_LOGITS && BLANK_ID < N_LOGITS);
            assert!(EOU_ID != EOB_ID && EOU_ID != BLANK_ID && EOB_ID != BLANK_ID);
            // Blank is the LAST logit; an off-by-one here silently turns every
            // "advance time" into a bogus emitted token.
            assert!(BLANK_ID == N_LOGITS - 1);
        };
    }

    #[test]
    fn vocab_line_parser_uses_line_index_as_id() {
        let dir = std::env::temp_dir().join("szca_eou_vocab_test");
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("vocab.txt");
        std::fs::write(&p, "<unk>\n\u{2581}hello\n\u{2581}world\n<EOU>\n").unwrap();
        let v = load_vocab_lines(p.to_str().unwrap()).unwrap();
        assert_eq!(v.len(), 4);
        assert_eq!(v[0], "<unk>");
        assert_eq!(v[1], "\u{2581}hello");
        assert_eq!(v[3], "<EOU>");
        std::fs::remove_file(&p).ok();
    }

    #[test]
    fn vocab_missing_file_errors() {
        assert!(load_vocab_lines("/nonexistent/szca/vocab.txt").is_err());
    }

    #[test]
    fn chunk_geometry_is_one_point_two_eight_seconds() {
        // 128 frames * 10 ms hop = 1.28 s of audio per encoder call.
        assert_eq!(CHUNK_FRAMES * crate::rt_stt_mel::HOP, 20_480);
        let ms = CHUNK_FRAMES * crate::rt_stt_mel::HOP * 1000
            / crate::rt_stt_mel::SAMPLE_RATE as usize;
        assert_eq!(ms, 1280);
    }
}
