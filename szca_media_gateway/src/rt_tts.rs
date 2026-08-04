/// Real Kokoro TTS stage: streaming ONNX inference with Misaki G2P.
///
/// Faithfully ports the correctness oracle from Python
/// (`szca_inference_service/app/tts.py`), using the pure-Rust `misaki-rs` port
/// of the Misaki G2P engine — the SAME algorithm the oracle uses
/// (`misaki.en.G2P(british=False, fallback=None)`).
///
/// Why Misaki and not raw espeak-ng: Kokoro was trained on Misaki's phoneme
/// alphabet, so Misaki reproduces the exact tokens the model expects. Feeding
/// raw espeak IPA through Kokoro's tokenizer would mispronounce words and drop
/// symbols that never map to the vocab. `misaki-rs` is self-contained pure Rust
/// (no C FFI, no `build.rs` link step, no global-state concurrency hazard).
///
/// Pipeline:
/// ```text
/// Text (LLM output)
///   → Misaki G2P                     (grapheme→phoneme, Kokoro alphabet)
///   → phoneme char → phoneme ID      (Kokoro tokenizer vocab)
///   → Kokoro ONNX forward            (input_ids + style + speed → 24 kHz waveform)
///   → 24 kHz → 16 kHz resample       (linear interpolation, matches session rate)
///   → PCM16 mono streaming deltas
/// ```
///
/// Model files expected in `TTS_MODEL_DIR` (default `./models/tts`):
///   `kokoro_v1.0_quantized.onnx`    — ONNX graph (input_ids, style, speed → waveform)
///   `kokoro_tokenizer.json`          — phoneme char → phoneme ID vocab
///   `kokoro_voices/<voice>.bin`      — voice style vectors [510, 1, 256] f32
///
/// Voice selection: `TTS_VOICE` env var (default `af_heart`). The per-call
/// `voice` override reloads the matching voice pack.
///
/// Language note: `misaki-rs` 0.3.0 supports English only (US/GB). Spanish is
/// not yet available in the pure-Rust port, so G2P runs as English (US),
/// matching the oracle. A future Spanish path would need a separate engine.

use std::collections::HashMap;
use std::sync::atomic::AtomicBool;

use misaki_rs::{G2P, Language};
use ndarray::{Array1, Array2};
use ort::session::Session;
use ort::value::Tensor;

use crate::onnx::init_ort;
use crate::rt_pipeline::TtsStage;
use crate::stage_pool::Replica;

// ---------------------------------------------------------------------------
// Phoneme → phoneme ID mapping (from kokoro_tokenizer.json vocab)
// ---------------------------------------------------------------------------

/// Build the phoneme char → phoneme ID mapping from the Kokoro tokenizer vocab.
/// The vocab maps single phoneme-alphabet characters to integer IDs.
fn build_phoneme_vocab(tokenizer_path: &str) -> Result<HashMap<char, i64>, String> {
    let tok = load_json(tokenizer_path)?;
    let vocab = tok["model"]["vocab"]
        .as_object()
        .ok_or("kokoro_tokenizer.json: missing model.vocab")?;

    let mut map = HashMap::with_capacity(vocab.len());
    for (ch_str, id_val) in vocab {
        if let Some(id) = id_val.as_i64() {
            // The vocab keys are single-character strings (phoneme symbols).
            // Special tokens like "$" (BOS/EOS = id 0) are included.
            if let Some(ch) = ch_str.chars().next() {
                map.insert(ch, id);
            }
        }
    }
    Ok(map)
}

/// Convert a Misaki phoneme string to phoneme IDs using the Kokoro vocab.
///
/// Mirrors the oracle's `_phonemes_to_ids`: prepend/append `$` (id 0) as
/// BOS/EOS and skip any symbol not in the vocab (Kokoro's normalizer strips
/// them rather than mapping to `<unk>`).
fn phonemes_to_ids(phonemes: &str, vocab: &HashMap<char, i64>) -> Vec<i64> {
    let mut ids = vec![0i64]; // leading BOS ($ = id 0)
    for ch in phonemes.chars() {
        if let Some(&id) = vocab.get(&ch) {
            ids.push(id);
        }
        // Unknown symbols dropped, matching Kokoro's normalizer behavior.
    }
    ids.push(0); // trailing EOS ($ = id 0)
    ids
}

// ---------------------------------------------------------------------------
// Voice pack loading
// ---------------------------------------------------------------------------

/// Voice pack: [510, 1, 256] f32 style vectors, indexed by phoneme-length.
const MAX_STYLE_ROWS: usize = 510;
const STYLE_DIM: usize = 256;

fn load_voice_pack(voice_path: &str) -> Result<Vec<f32>, String> {
    let data = std::fs::read(voice_path).map_err(|e| format!("read voice {voice_path}: {e}"))?;
    let floats: Vec<f32> = data
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    if floats.len() < MAX_STYLE_ROWS * STYLE_DIM {
        return Err(format!(
            "voice pack too small: {} floats, expected >= {}",
            floats.len(),
            MAX_STYLE_ROWS * STYLE_DIM
        ));
    }
    Ok(floats)
}

/// Extract a style vector for a given token count from the voice pack.
/// `n` is the number of input tokens (clamped to [0, MAX_STYLE_ROWS - 1]).
fn style_for_length(voice_pack: &[f32], n: usize) -> Vec<f32> {
    let row = n.min(MAX_STYLE_ROWS - 1);
    let offset = row * STYLE_DIM;
    voice_pack[offset..offset + STYLE_DIM].to_vec()
}

// ---------------------------------------------------------------------------
// Linear resample 24 kHz → 16 kHz
// ---------------------------------------------------------------------------

fn resample_24k_to_16k(input: &[f32]) -> Vec<f32> {
    if input.is_empty() {
        return Vec::new();
    }
    // 16/24 = 2/3 ratio
    let out_len = (input.len() * 16) / 24;
    let mut output = Vec::with_capacity(out_len);
    for i in 0..out_len {
        let src_pos = (i as f64) * (24.0 / 16.0);
        let idx = src_pos as usize;
        let frac = src_pos - idx as f64;
        let s0 = input[idx.min(input.len() - 1)];
        let s1 = input[(idx + 1).min(input.len() - 1)];
        output.push(s0 + (s1 - s0) * frac as f32);
    }
    output
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Loaded Kokoro TTS engine (ONNX session + Misaki G2P + voice pack).
pub struct KokoroTts {
    session: Session,
    g2p: G2P,
    phoneme_vocab: HashMap<char, i64>,
    voice_pack: Vec<f32>,
    /// Directory holding `kokoro_voices/<voice>.bin`, used to reload the voice
    /// pack when a per-call voice override is requested.
    model_dir: String,
    voice: String,
}

impl KokoroTts {
    /// Load Kokoro TTS from environment configuration.
    pub fn from_env() -> Result<Self, String> {
        let dir = std::env::var("TTS_MODEL_DIR")
            .ok()
            .filter(|d| !d.is_empty())
            .unwrap_or_else(|| "./models/tts".to_string());
        Self::load(&dir)
    }

    fn load(dir: &str) -> Result<Self, String> {
        init_ort()?;

        // Misaki G2P — English (US), no espeak fallback, matching the oracle
        // (`misaki.en.G2P(british=False, fallback=None)`). Constructed once;
        // loading the POS tagger + lexicon is not free, so we reuse it.
        let g2p = G2P::new(Language::EnglishUS);

        // Load ONNX session.
        let kokoro_path = format!("{dir}/kokoro_v1.0_quantized.onnx");
        let session = Session::builder()
            .map_err(|e| format!("session builder: {e}"))?
            .commit_from_file(&kokoro_path)
            .map_err(|e| format!("load kokoro ONNX: {e}"))?;

        // Load phoneme vocab.
        let tokenizer_path = format!("{dir}/kokoro_tokenizer.json");
        let phoneme_vocab = build_phoneme_vocab(&tokenizer_path)?;

        // Load voice pack.
        let voice = std::env::var("TTS_VOICE")
            .ok()
            .filter(|v| !v.is_empty())
            .unwrap_or_else(|| "af_heart".to_string());
        let voice_path = format!("{dir}/kokoro_voices/{voice}.bin");
        let voice_pack = load_voice_pack(&voice_path)?;

        tracing::info!(
            model = %kokoro_path,
            voice = %voice,
            vocab_entries = phoneme_vocab.len(),
            "Kokoro TTS loaded (Misaki G2P, pure Rust)"
        );

        Ok(Self {
            session,
            g2p,
            phoneme_vocab,
            voice_pack,
            model_dir: dir.to_string(),
            voice,
        })
    }

    /// Synthesize text → 24kHz f32 waveform via Misaki G2P + Kokoro ONNX.
    fn synthesize_to_24k(&mut self, text: &str) -> Result<Vec<f32>, String> {
        let text = text.trim();
        if text.is_empty() {
            return Ok(Vec::new());
        }

        // Step 1: Text → phonemes via Misaki (Kokoro's phoneme alphabet).
        let (phonemes, _tokens) = self
            .g2p
            .g2p(text)
            .map_err(|e| format!("misaki g2p: {e}"))?;
        if phonemes.is_empty() {
            return Ok(Vec::new());
        }
        tracing::debug!(phonemes = %phonemes, "Misaki G2P output");

        // Step 2: phonemes → phoneme IDs via Kokoro vocab.
        let input_ids = phonemes_to_ids(&phonemes, &self.phoneme_vocab);
        let n_tokens = input_ids.len().saturating_sub(2); // exclude BOS/EOS

        // Step 3: Get style vector from voice pack.
        let style = style_for_length(&self.voice_pack, n_tokens);

        // Step 4: Run Kokoro ONNX forward pass.
        let ids_arr = Array2::from_shape_vec((1, input_ids.len()), input_ids)
            .map_err(|e| format!("input_ids shape: {e}"))?;
        let style_arr =
            Array2::from_shape_vec((1, STYLE_DIM), style).map_err(|e| format!("style shape: {e}"))?;
        let speed_arr = Array1::from_vec(vec![1.0f32]);

        let outputs = self
            .session
            .run(ort::inputs![
                "input_ids" => Tensor::from_array(ids_arr).map_err(|e| format!("input_ids tensor: {e}"))?,
                "style" => Tensor::from_array(style_arr).map_err(|e| format!("style tensor: {e}"))?,
                "speed" => Tensor::from_array(speed_arr).map_err(|e| format!("speed tensor: {e}"))?,
            ])
            .map_err(|e| format!("kokoro forward: {e}"))?;

        let (_, waveform) = outputs[0]
            .try_extract_tensor::<f32>()
            .map_err(|e| format!("waveform extract: {e}"))?;

        Ok(waveform.to_vec())
    }

    /// Core synthesis path: runs the Kokoro model and streams PCM16 chunks.
    /// Does NOT manage voice packs — caller handles voice swap/restore.
    fn synthesize_pcm(
        &mut self,
        text: &str,
        cancel: &std::sync::atomic::AtomicBool,
        on_audio: &mut dyn FnMut(&[u8]),
    ) {
        use std::sync::atomic::Ordering;

        match self.synthesize_to_24k(text) {
            Ok(waveform_24k) => {
                let waveform_16k = resample_24k_to_16k(&waveform_24k);

                if cancel.load(Ordering::Relaxed) {
                    tracing::debug!("Kokoro TTS: cancelled before output");
                } else {
                    // Stream PCM16 chunks (20 ms = 320 samples @ 16 kHz = 640 bytes).
                    const CHUNK_SAMPLES: usize = 320;
                    for chunk in waveform_16k.chunks(CHUNK_SAMPLES) {
                        if cancel.load(Ordering::Relaxed) {
                            break;
                        }
                        let mut buf = Vec::with_capacity(chunk.len() * 2);
                        for &sample in chunk {
                            let pcm16 = (sample.clamp(-1.0, 1.0) * 32767.0) as i16;
                            buf.extend_from_slice(&pcm16.to_le_bytes());
                        }
                        on_audio(&buf);
                    }
                }
            }
            Err(e) => {
                tracing::error!(error = %e, "Kokoro TTS synthesis failed");
            }
        }
    }
}

impl TtsStage for KokoroTts {
    fn synthesize(
        &mut self,
        text: &str,
        voice: Option<&str>,
        cancel: &std::sync::atomic::AtomicBool,
        on_audio: &mut dyn FnMut(&[u8]),
    ) {
        // Per-call voice override: reload the voice pack for the requested voice
        // and restore the default afterwards.
        let mut restore: Option<Vec<f32>> = None;
        if let Some(v) = voice {
            if v != self.voice {
                let path = format!("{}/kokoro_voices/{}.bin", self.model_dir, v);
                match load_voice_pack(&path) {
                    Ok(pack) => restore = Some(std::mem::replace(&mut self.voice_pack, pack)),
                    Err(e) => tracing::warn!(voice = %v, error = %e,
                        "voice override failed to load; using default voice"),
                }
            }
        }

        self.synthesize_pcm(text, cancel, on_audio);

        // Restore the default voice pack if we swapped it in.
        if let Some(pack) = restore {
            self.voice_pack = pack;
        }
    }
}

// ---------------------------------------------------------------------------
// StagePool Replica implementation
// ---------------------------------------------------------------------------

/// Input for the TTS pool: text to synthesize plus an optional voice override.
pub struct TtsInput {
    pub text: String,
    pub voice: Option<String>,
}

impl Replica for KokoroTts {
    type Input = TtsInput;
    type Delta = Vec<u8>;
    type Output = ();

    fn process(
        &mut self,
        input: Self::Input,
        cancel: &AtomicBool,
        emit: &mut dyn FnMut(Self::Delta),
    ) -> Self::Output {
        use std::sync::atomic::Ordering;

        // Per-job voice override: swap voice pack, synthesize, restore.
        let mut restore: Option<Vec<f32>> = None;
        if let Some(ref v) = input.voice {
            if v != &self.voice {
                let path = format!("{}/kokoro_voices/{}.bin", self.model_dir, v);
                match load_voice_pack(&path) {
                    Ok(pack) => restore = Some(std::mem::replace(&mut self.voice_pack, pack)),
                    Err(e) => tracing::warn!(voice = %v, error = %e,
                        "TTS pool voice override failed; using default"),
                }
            }
        }

        // Wrap emit to check cancel cooperatively.
        let mut emit_checked = |chunk: &[u8]| {
            if cancel.load(Ordering::Relaxed) {
                return;
            }
            emit(chunk.to_vec());
        };

        self.synthesize_pcm(&input.text, cancel, &mut emit_checked);

        // Restore the default voice pack if we swapped.
        if let Some(pack) = restore {
            self.voice_pack = pack;
        }
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn load_json(path: &str) -> Result<serde_json::Value, String> {
    let content = std::fs::read_to_string(path).map_err(|e| format!("read {path}: {e}"))?;
    serde_json::from_str(&content).map_err(|e| format!("parse JSON {path}: {e}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn phonemes_to_ids_maps_known_chars() {
        let mut vocab = HashMap::new();
        vocab.insert('h', 50i64);
        vocab.insert('e', 47);
        vocab.insert('l', 54);
        vocab.insert('o', 57);
        vocab.insert(' ', 16);
        vocab.insert('$', 0);

        // "hel lo" → BOS, h, e, l, space, l, o, EOS
        let ids = phonemes_to_ids("hel lo", &vocab);
        assert_eq!(ids, vec![0, 50, 47, 54, 16, 54, 57, 0]);
    }

    #[test]
    fn phonemes_to_ids_drops_unknown_chars() {
        let mut vocab = HashMap::new();
        vocab.insert('a', 43i64);
        // Stress mark ˈ not in vocab → should be dropped.
        let ids = phonemes_to_ids("aˈb", &vocab);
        assert_eq!(ids, vec![0, 43, 0]); // ˈ dropped, b dropped
    }

    #[test]
    fn resample_preserves_length_ratio() {
        let input = vec![0.0f32; 2400]; // 100 ms @ 24 kHz
        let output = resample_24k_to_16k(&input);
        // Should be ~1600 samples (100 ms @ 16 kHz)
        assert!((1595..1605).contains(&output.len()));
    }

    #[test]
    fn style_for_length_clamps_to_max() {
        let pack = vec![0.0f32; MAX_STYLE_ROWS * STYLE_DIM];
        let s = style_for_length(&pack, 999);
        assert_eq!(s.len(), STYLE_DIM);
    }

    #[test]
    fn load_json_parses_tokenizer() {
        // Use the real tokenizer file if it exists.
        let path = std::env::current_dir()
            .unwrap()
            .join("../models/tts/kokoro_tokenizer.json");
        if path.exists() {
            let val = load_json(path.to_str().unwrap());
            assert!(val.is_ok());
            let obj = val.unwrap();
            let vocab = obj["model"]["vocab"].as_object();
            assert!(vocab.is_some());
            assert!(vocab.unwrap().len() > 100);
        }
    }

    #[test]
    fn misaki_g2p_produces_kokoro_phonemes() {
        // Misaki should phonemize common English without the unknown marker,
        // and the phonemes should map into the Kokoro vocab.
        let g2p = G2P::new(Language::EnglishUS);
        let (phonemes, _) = g2p.g2p("hello world").unwrap();
        assert!(!phonemes.is_empty());
        assert!(!phonemes.contains('❓'), "unexpected unknown marker: {phonemes}");
    }
}
