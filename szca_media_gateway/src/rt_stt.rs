/// Real Parakeet TDT 0.6B V3 (int8 ONNX) speech-to-text stage.
///
/// Faithful Rust port of `szca_inference_service/app/stt.py` (the correctness
/// oracle). Runs three chained ONNX graphs:
///
/// ```text
/// PCM16 mono 16 kHz
///   → parakeet_nemo128.onnx   (log-mel feature frontend, exact NeMo features)
///   → parakeet_encoder.int8   (Conformer acoustic encoder, 8× subsampling)
///   → parakeet_decoder_joint  (TDT greedy decode: token + duration per step)
///   → SentencePiece vocab      (detokenize, '_' → space)
/// ```
///
/// The frontend is a real ONNX graph rather than hand-ported DSP, so NeMo's
/// exact mel features are reproduced bit-for-bit — the usual "runs but
/// transcribes garbage" trap is avoided.
///
/// The decoder is a Token-and-Duration Transducer: each joint step emits a
/// token logit block AND a duration logit block; the duration tells us how
/// many encoder frames to skip, which is what makes TDT fast.
///
/// Geometry (from the oracle):
///   vocab_size = 8193, blank_id = 8192, num_durations = 5,
///   pred_state_dim = 640, pred_state_layers = 2.

use std::sync::atomic::AtomicBool;

use ndarray::{Array1, Array2, Array3};
use ort::session::Session;
use ort::value::Tensor;

use crate::onnx::init_ort;
use crate::rt_pipeline::SttStage;
use crate::stage_pool::Replica;

// ---------------------------------------------------------------------------
// TDT / vocab constants (derived from the model + vocab.txt)
// ---------------------------------------------------------------------------

/// Number of entries in the SentencePiece vocabulary.
const VOCAB_SIZE: usize = 8193;

/// Blank token id (last entry in vocab.txt).
const BLANK_ID: usize = 8192;

/// Number of TDT duration bins: [0, 1, 2, 3, 4] frames.
const NUM_DURATIONS: usize = 5;

/// Width of the prediction-network LSTM hidden state.
const PRED_STATE_DIM: usize = 640;

/// Number of stacked LSTM layers in the prediction network.
const PRED_STATE_LAYERS: usize = 2;

/// Safety cap on tokens emitted per encoder frame before we force a time step.
const MAX_SYMBOLS_PER_STEP: usize = 10;

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Loaded Parakeet TDT STT engine (three chained ONNX sessions + vocab).
pub struct ParakeetStt {
    mel: Session,
    encoder: Session,
    decoder: Session,
    vocab: Vec<String>,
}

impl ParakeetStt {
    /// Load the Parakeet STT engine using environment configuration:
    ///   * `STT_MODEL_DIR` — model directory (default `./models/stt`)
    ///
    /// Expected files in the directory:
    ///   `parakeet_nemo128.onnx`, `parakeet_encoder.int8.onnx`,
    ///   `parakeet_decoder_joint.int8.onnx`, `parakeet_vocab.txt`.
    pub fn from_env() -> Result<Self, String> {
        let dir = std::env::var("STT_MODEL_DIR")
            .ok()
            .filter(|d| !d.is_empty())
            .unwrap_or_else(|| "./models/stt".to_string());
        Self::load(&dir)
    }

    fn load(dir: &str) -> Result<Self, String> {
        init_ort()?;

        let mel_path = format!("{dir}/parakeet_nemo128.onnx");
        let enc_path = format!("{dir}/parakeet_encoder.int8.onnx");
        let dec_path = format!("{dir}/parakeet_decoder_joint.int8.onnx");
        let vocab_path = format!("{dir}/parakeet_vocab.txt");

        let build = |p: &str| -> Result<Session, String> {
            let session = Session::builder()
                .map_err(|e| format!("session builder: {e}"))?
                .commit_from_file(p)
                .map_err(|e| format!("load {p}: {e}"))?;
            Ok(session)
        };

        let mel = build(&mel_path)?;
        let encoder = build(&enc_path)?;
        let decoder = build(&dec_path)?;
        let vocab = load_vocab(&vocab_path)?;

        tracing::info!(
            vocab_entries = vocab.len(),
            "Parakeet TDT STT loaded from {dir}"
        );

        Ok(Self {
            mel,
            encoder,
            decoder,
            vocab,
        })
    }

    /// Run the full three-graph pipeline on a 16 kHz mono f32 waveform.
    ///
    /// `partial` receives the growing transcript as the TDT decode commits
    /// tokens, so the client sees real interim hypotheses (not a placeholder).
    ///
    /// NOTE: the *input* here is the complete utterance — our Conformer encoder
    /// is full-context (full-sequence attention + 8x subsampling), so it cannot
    /// consume audio incrementally. True streaming input would require a
    /// cache-aware ("chunked") Parakeet export with exported cache states; that
    /// is a model change, not a code change, so we don't fake it. The interim
    /// output below is genuine — it reflects the decode as it progresses.
    fn run_pipeline(
        &mut self,
        waveform: &[f32],
        partial: &mut dyn FnMut(&str),
    ) -> Result<String, String> {
        if waveform.is_empty() {
            return Ok(String::new());
        }

        // 1. Log-mel features [1, 128, T] via nemo128 frontend.
        //    Scoped so the borrow on self.mel ends before we touch self.encoder.
        let (feats, feat_lens) = {
            let wav = Array2::from_shape_vec((1, waveform.len()), waveform.to_vec())
                .map_err(|e| format!("wav shape: {e}"))?;
            let wav_len = Array1::from_vec(vec![waveform.len() as i64]);

            let mel_out = self
                .mel
                .run(ort::inputs![
                    "waveforms" => Tensor::from_array(wav).map_err(|e| format!("wav tensor: {e}"))?,
                    "waveforms_lens" => Tensor::from_array(wav_len).map_err(|e| format!("wav_len tensor: {e}"))?,
                ])
                .map_err(|e| format!("mel forward: {e}"))?;

            let (feat_shape, feat_data) = mel_out[0]
                .try_extract_tensor::<f32>()
                .map_err(|e| format!("mel output: {e}"))?;
            let feat_dims: Vec<usize> = feat_shape.iter().map(|&d| d as usize).collect();
            let feats = Array3::from_shape_vec(
                (feat_dims[0], feat_dims[1], feat_dims[2]),
                feat_data.to_vec(),
            )
            .map_err(|e| format!("feats reshape: {e}"))?;

            let (_, feat_lens_data) = mel_out[1]
                .try_extract_tensor::<i64>()
                .map_err(|e| format!("feat_lens: {e}"))?;
            let feat_lens = Array1::from_vec(feat_lens_data.to_vec());

            (feats, feat_lens)
        };

        // 2. Conformer encoder → [1, 1024, T'] where T' ≈ T/8.
        //    Scoped so the borrow on self.encoder ends before we touch self.decoder.
        let (enc_out_arr, num_frames) = {
            let enc_out = self
                .encoder
                .run(ort::inputs![
                    "audio_signal" => Tensor::from_array(feats).map_err(|e| format!("feats tensor: {e}"))?,
                    "length" => Tensor::from_array(feat_lens).map_err(|e| format!("feat_lens tensor: {e}"))?,
                ])
                .map_err(|e| format!("encoder forward: {e}"))?;

            let (enc_shape, enc_data) = enc_out[0]
                .try_extract_tensor::<f32>()
                .map_err(|e| format!("enc output: {e}"))?;
            let enc_dims: Vec<usize> = enc_shape.iter().map(|&d| d as usize).collect();
            let enc_out_arr = Array3::from_shape_vec(
                (enc_dims[0], enc_dims[1], enc_dims[2]),
                enc_data.to_vec(),
            )
            .map_err(|e| format!("enc reshape: {e}"))?;

            let (_, enc_lens_data) = enc_out[1]
                .try_extract_tensor::<i64>()
                .map_err(|e| format!("enc_lens: {e}"))?;
            let num_frames = *enc_lens_data.first().ok_or("enc_lens empty")? as usize;

            (enc_out_arr, num_frames)
        };

        // 3. TDT greedy decode over encoder frames, emitting interims as the
        //    hypothesis grows.
        let hyp = self.tdt_greedy(&enc_out_arr, num_frames, partial)?;
        Ok(self.detokenize(&hyp))
    }

    /// Token-and-Duration Transducer greedy decode.
    ///
    /// Mirrors the oracle's `transcribe` decode loop exactly:
    ///   - Prime with BLANK_ID
    ///   - At each frame, run the joint network for the current target
    ///   - If blank → advance time by max(duration, 1), break inner loop
    ///   - If real token → commit, update pred state, advance time by duration
    ///   - Cap at MAX_SYMBOLS_PER_STEP to avoid pathological loops
    fn tdt_greedy(
        &mut self,
        enc_out: &ndarray::Array3<f32>,
        num_frames: usize,
        partial: &mut dyn FnMut(&str),
    ) -> Result<Vec<usize>, String> {
        let mut state1 = zero_pred_state();
        let mut state2 = zero_pred_state();
        let mut last_token: i64 = BLANK_ID as i64;
        let mut hyp: Vec<usize> = Vec::new();

        let mut t: usize = 0;
        while t < num_frames {
            // Slice encoder output at frame t: [1, 1024, 1].
            let enc_frame = enc_out.slice(ndarray::s![.., .., t..t + 1]);
            let mut emitted: usize = 0;
            let mut advanced = false;

            while emitted < MAX_SYMBOLS_PER_STEP {
                let (tok_logits, dur_logits, s1, s2) =
                    self.decode_step(enc_frame.as_standard_layout().into_owned(), &last_token, &state1, &state2)?;

                let token = argmax_i64(&tok_logits);
                let duration = argmax_i64(&dur_logits) as usize;

                if token == BLANK_ID as i64 {
                    // Blank: advance time, do NOT update pred state.
                    t += duration.max(1);
                    advanced = true;
                    break;
                }

                // Real token: commit it, update state, jump duration frames.
                hyp.push(token as usize);
                last_token = token;
                state1 = s1;
                state2 = s2;
                emitted += 1;
                // Emit a real interim: the transcript decoded so far. We only
                // push when it starts a new word (SentencePiece '▁' marker) to
                // avoid mid-word flicker while still streaming as speech decodes.
                if self
                    .vocab
                    .get(token as usize)
                    .is_some_and(|p| p.starts_with('\u{2581}'))
                {
                    let interim = detokenize_ids(&self.vocab, &hyp);
                    if !interim.is_empty() {
                        partial(&interim);
                    }
                }
                if duration > 0 {
                    t += duration;
                    advanced = true;
                    break;
                }
            }

            if !advanced {
                t += 1; // force progress after MAX_SYMBOLS_PER_STEP at dur=0
            }
        }

        Ok(hyp)
    }

    /// Run the decoder+joint network for one target at a single encoder frame.
    ///
    /// Returns `(token_logits[8193], duration_logits[5], new_state1, new_state2)`.
    #[allow(clippy::type_complexity)]
    fn decode_step(
        &mut self,
        enc_frame: Array3<f32>,
        target: &i64,
        state1: &Array3<f32>,
        state2: &Array3<f32>,
    ) -> Result<(Vec<f32>, Vec<f32>, Array3<f32>, Array3<f32>), String> {
        // The graph declares `targets` as INT32 (verified against the export),
        // while token ids are i64 everywhere else in this decoder — vocab ids,
        // argmax results, BLANK_ID. Narrow only at the tensor boundary: passing
        // i64 makes ORT reject the call with "Unexpected input data type", which
        // the pool then reports as a failed pipeline and an EMPTY transcript.
        let targets = Array2::from_shape_vec((1, 1), vec![*target as i32])
            .map_err(|e| format!("targets shape: {e}"))?;
        let target_length = Array1::from_vec(vec![1i32]);

        let out = self
            .decoder
            .run(ort::inputs![
                "encoder_outputs" => Tensor::from_array(enc_frame).map_err(|e| format!("enc_frame tensor: {e}"))?,
                "targets" => Tensor::from_array(targets).map_err(|e| format!("targets tensor: {e}"))?,
                "target_length" => Tensor::from_array(target_length).map_err(|e| format!("target_len tensor: {e}"))?,
                "input_states_1" => Tensor::from_array(state1.clone()).map_err(|e| format!("state1 tensor: {e}"))?,
                "input_states_2" => Tensor::from_array(state2.clone()).map_err(|e| format!("state2 tensor: {e}"))?,
            ])
            .map_err(|e| format!("decoder forward: {e}"))?;

        // outputs: [logits, prednet_lengths, out_state_1, out_state_2]
        let (_, logits_data) = out[0]
            .try_extract_tensor::<f32>()
            .map_err(|e| format!("logits extract: {e}"))?;
        let logits: Vec<f32> = logits_data.to_vec();

        let token_logits: Vec<f32> = logits[..VOCAB_SIZE].to_vec();
        let duration_logits: Vec<f32> =
            logits[VOCAB_SIZE..VOCAB_SIZE + NUM_DURATIONS].to_vec();

        // Extract pred states, copying data out to avoid borrow conflicts.
        let new_state1 = {
            let (sh, sd) = out[2].try_extract_tensor::<f32>()
                .map_err(|e| format!("state1 extract: {e}"))?;
            let dims: Vec<usize> = sh.iter().map(|&d| d as usize).collect();
            Array3::from_shape_vec((dims[0], dims[1], dims[2]), sd.to_vec())
                .map_err(|e| format!("state1 reshape: {e}"))?
        };
        let new_state2 = {
            let (sh, sd) = out[3].try_extract_tensor::<f32>()
                .map_err(|e| format!("state2 extract: {e}"))?;
            let dims: Vec<usize> = sh.iter().map(|&d| d as usize).collect();
            Array3::from_shape_vec((dims[0], dims[1], dims[2]), sd.to_vec())
                .map_err(|e| format!("state2 reshape: {e}"))?
        };

        Ok((token_logits, duration_logits, new_state1, new_state2))
    }

    /// Detokenize a sequence of token IDs using SentencePiece-style vocab.
    ///
    /// `'_'` (U+2581) marks a leading space, consistent with the NeMo tokenizer.
    fn detokenize(&self, ids: &[usize]) -> String {
        detokenize_ids(&self.vocab, ids)
    }
}

impl SttStage for ParakeetStt {
    fn transcribe(&mut self, pcm: &[u8], partial: &mut dyn FnMut(&str)) -> String {
        // Convert PCM16 (little-endian) → f32 in [-1, 1].
        let samples: Vec<f32> = pcm
            .chunks_exact(2)
            .map(|c| i16::from_le_bytes([c[0], c[1]]) as f32 / 32768.0)
            .collect();

        match self.run_pipeline(&samples, partial) {
            Ok(text) => text,
            Err(e) => {
                tracing::error!(error = %e, "STT pipeline failed");
                String::new()
            }
        }
    }
}

// ---------------------------------------------------------------------------
// StagePool Replica implementation
// ---------------------------------------------------------------------------

/// Input for the STT pool: raw PCM16 mono 16 kHz audio.
pub struct SttInput {
    pub pcm: Vec<u8>,
}

impl Replica for ParakeetStt {
    type Input = SttInput;
    type Delta = String;
    type Output = String;

    fn process(
        &mut self,
        input: Self::Input,
        cancel: &AtomicBool,
        emit: &mut dyn FnMut(Self::Delta),
    ) -> Self::Output {
        use std::sync::atomic::Ordering;

        // Convert PCM16 (little-endian) → f32 in [-1, 1].
        let samples: Vec<f32> = input
            .pcm
            .chunks_exact(2)
            .map(|c| i16::from_le_bytes([c[0], c[1]]) as f32 / 32768.0)
            .collect();

        // Wrap emit to check cancel cooperatively — if the caller set the
        // cancel flag mid-stream we stop emitting and return early.
        let mut emit_checked = |text: &str| {
            if cancel.load(Ordering::Relaxed) {
                return;
            }
            emit(text.to_string());
        };

        match self.run_pipeline(&samples, &mut emit_checked) {
            Ok(text) => text,
            Err(e) => {
                tracing::error!(error = %e, "STT pool pipeline failed");
                String::new()
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Zero-initialised prediction-network state: `[2, 1, 640]` f32.
fn zero_pred_state() -> Array3<f32> {
    Array3::<f32>::zeros((PRED_STATE_LAYERS, 1, PRED_STATE_DIM))
}

/// Argmax over a small slice, returning the index as i64.
fn argmax_i64(vals: &[f32]) -> i64 {
    vals.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i as i64)
        .unwrap_or(0)
}

/// Detokenize token IDs using a SentencePiece-style vocab.
///
/// `'_'` (U+2581) marks a leading space, consistent with the NeMo tokenizer.
fn detokenize_ids(vocab: &[String], ids: &[usize]) -> String {
    let mut pieces = Vec::with_capacity(ids.len());
    for &id in ids {
        if id < vocab.len() {
            pieces.push(vocab[id].as_str());
        }
    }
    // SentencePiece uses U+2581 (LOWER ONE EIGHTH BLOCK) to mark leading spaces.
    pieces.join("").replace('\u{2581}', " ").trim().to_string()
}

/// Parse `token id` lines into an index-ordered vector.
///
/// Format: `<token><space><id>` (split from the right to handle tokens with
/// internal spaces).
fn load_vocab(path: &str) -> Result<Vec<String>, String> {
    let text = std::fs::read_to_string(path)
        .map_err(|e| format!("read vocab {path}: {e}"))?;
    let mut table: std::collections::HashMap<i64, String> = std::collections::HashMap::new();
    let mut max_id: i64 = -1;
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        // Split from the right: "▁hello 123" → ("▁hello", " ", "123")
        if let Some(last_space) = line.rfind(' ') {
            let piece = &line[..last_space];
            let id_str = &line[last_space + 1..];
            if let Ok(id) = id_str.parse::<i64>() {
                table.insert(id, piece.to_string());
                max_id = max_id.max(id);
            }
        }
    }
    if max_id < 0 {
        return Err(format!("vocab file {path} has no valid entries"));
    }
    let mut vocab = Vec::with_capacity((max_id + 1) as usize);
    for i in 0..=max_id {
        vocab.push(table.get(&i).cloned().unwrap_or_default());
    }
    Ok(vocab)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn argmax_i64_basic() {
        assert_eq!(argmax_i64(&[0.1, 0.9, 0.5]), 1);
        assert_eq!(argmax_i64(&[3.0, 1.0, 2.0]), 0);
        assert_eq!(argmax_i64(&[-1.0, -2.0, -0.5]), 2);
    }

    #[test]
    fn zero_pred_state_shape() {
        let s = zero_pred_state();
        assert_eq!(s.dim(), (PRED_STATE_LAYERS, 1, PRED_STATE_DIM));
    }

    #[test]
    fn detokenize_sentencepiece_style() {
        let vocab = vec![
            "<unk>".into(),
            "▁Hello".into(),
            "▁".into(),
            "world".into(),
        ];
        // IDs [1, 2, 3] → "▁Hello" + "▁" + "world" → " Hello world" → trim → "Hello world"
        assert_eq!(detokenize_ids(&vocab, &[1, 2, 3]), "Hello world");
    }
}
