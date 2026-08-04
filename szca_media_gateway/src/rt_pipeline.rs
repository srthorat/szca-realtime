/// Realtime STT -> LLM -> TTS pipeline behind swappable stage traits.
///
/// Production stages are ONNX-backed (`ParakeetStt`, `ParakeetEouStt`,
/// `SherpaZipformer`, `QwenLlm` / `VllmClient`, `KokoroTts`) and are wired
/// through [`crate::stage_pools::StagePools`] in the gateway. Stub stages remain
/// for tests and weight-free CI.
///
/// Streaming contract:
///   * STT consumes accumulated utterance PCM and yields incremental transcript
///     text, then a final string.
///   * LLM consumes a prompt and yields text token-chunks via a callback until
///     done or cancelled.
///   * TTS consumes a text clause and yields PCM16 audio chunks via a callback.
///
/// Cancellation is cooperative: long-running stages check a `&AtomicBool` and
/// stop early when it is set (barge-in).

use std::sync::atomic::{AtomicBool, Ordering};

/// Audio sample rate the pipeline produces/consumes (Hz).
pub const PIPELINE_SAMPLE_RATE: u32 = 16_000;

/// Max bytes per realtime `AppendAudio` chunk (matches HTTP body cap).
pub const MAX_AUDIO_CHUNK_BYTES: usize = 1024 * 1024;

/// Validate inbound PCM16 mono audio at the session boundary.
///
/// Empty chunks are allowed. Odd byte length is rejected (invalid PCM16).
pub fn validate_pcm16_chunk(pcm: &[u8]) -> Result<(), String> {
    if pcm.is_empty() {
        return Ok(());
    }
    if pcm.len() % 2 != 0 {
        return Err("audio must be even-length PCM16 (16-bit sample alignment)".into());
    }
    if pcm.len() > MAX_AUDIO_CHUNK_BYTES {
        return Err(format!(
            "audio chunk exceeds {} byte limit",
            MAX_AUDIO_CHUNK_BYTES
        ));
    }
    Ok(())
}

/// Output of processing an incremental audio chunk in a streaming STT stage.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct SttChunkResult {
    /// Newly decoded text for this chunk (may be empty).
    pub delta_text: String,
    /// The model signalled end-of-utterance — turn can end immediately.
    pub end_of_utterance: bool,
}

/// Speech-to-text stage.
pub trait SttStage: Send {
    /// Transcribe a complete utterance (PCM16 mono 16 kHz).
    ///
    /// `partial` is invoked with incremental hypotheses as decoding proceeds;
    /// the returned String is the final transcript.
    fn transcribe(&mut self, pcm: &[u8], partial: &mut dyn FnMut(&str)) -> String;

    /// Process an incremental audio chunk for streaming backends.
    ///
    /// Returns `Some(SttChunkResult)` if the stage supports streaming and a chunk
    /// completed; returns `None` for batch stages or partial audio.
    fn push_chunk(&mut self, _pcm: &[u8]) -> Option<SttChunkResult> {
        None
    }

    /// Reset internal streaming state between turns.
    fn reset_stream(&mut self) {}

    /// Whether this stage supports lookback augmentation to fix VAD timing windows.
    ///
    /// Returns true if the stage can accept augmented audio that includes pre-speech
    /// segments to improve transcript completeness when VAD misses the exact start
    /// of speech.
    fn supports_lookback(&self) -> bool {
        false
    }
}

/// Large-language-model stage.
pub trait LlmStage: Send {
    /// Generate a reply to `prompt`, streaming text chunks via `on_token`.
    ///
    /// Must return early (with whatever was produced so far) when `cancel` is
    /// set. Returns the full concatenated reply text.
    fn generate(
        &mut self,
        prompt: &str,
        instructions: Option<&str>,
        cancel: &AtomicBool,
        on_token: &mut dyn FnMut(&str),
    ) -> String;
}

/// Text-to-speech stage.
pub trait TtsStage: Send {
    /// Synthesize `text` into PCM16 mono 16 kHz audio, streaming chunks via
    /// `on_audio`. Must stop early when `cancel` is set.
    fn synthesize(
        &mut self,
        text: &str,
        voice: Option<&str>,
        cancel: &AtomicBool,
        on_audio: &mut dyn FnMut(&[u8]),
    );
}

/// The full pipeline: owns one instance of each stage.
pub struct Pipeline {
    pub stt: Box<dyn SttStage>,
    pub llm: Box<dyn LlmStage>,
    pub tts: Box<dyn TtsStage>,
}

impl Pipeline {
    /// Construct the Phase-1 stubbed pipeline.
    pub fn stubbed() -> Self {
        Self {
            stt: Box::new(StubStt),
            llm: Box::new(StubLlm),
            tts: Box::new(StubTts),
        }
    }

    /// Construct the pipeline with real model stages where provisioned,
    /// falling back to stubs for any missing stage (logging why).
    ///
    /// This keeps the gateway runnable without model weights while letting a
    /// properly provisioned deployment stream genuine STT/LLM/TTS over WS.
    pub fn with_real_models() -> Self {
        // Honour STT_BACKEND here too. This is the no-pool fallback path, and if
        // it ignored the env var the pooled and unpooled paths would silently run
        // different acoustic models against the same config.
        use crate::stage_pools::{dev_model_selection, SttModel};
        let stt: Box<dyn SttStage> = match dev_model_selection() {
            SttModel::Zipformer => {
                match crate::rt_stt_zipformer::SherpaZipformer::from_env() {
                    Ok(model) => {
                        tracing::info!("Realtime pipeline using real Sherpa Zipformer STT stage");
                        Box::new(model)
                    }
                    Err(e) => {
                        tracing::warn!(error = %e, "Zipformer STT unavailable; using stub STT stage");
                        Box::new(StubStt)
                    }
                }
            }
            SttModel::Streaming => {
                match crate::rt_stt_eou::ParakeetEouStt::from_env() {
                    Ok(model) => {
                        tracing::info!("Realtime pipeline using real streaming EOU STT stage");
                        Box::new(model)
                    }
                    Err(e) => {
                        tracing::warn!(error = %e, "Streaming STT unavailable; using stub STT stage");
                        Box::new(StubStt)
                    }
                }
            }
            SttModel::Parakeet => {
                match crate::rt_stt::ParakeetStt::from_env() {
                    Ok(model) => {
                        tracing::info!("Realtime pipeline using real Parakeet STT stage");
                        Box::new(model)
                    }
                    Err(e) => {
                        tracing::warn!(error = %e, "Real STT unavailable; using stub STT stage");
                        Box::new(StubStt)
                    }
                }
            }
        };
        let llm: Box<dyn LlmStage> = match crate::rt_llm::QwenLlm::from_env() {
            Ok(model) => {
                tracing::info!("Realtime pipeline using real Qwen LLM stage");
                Box::new(model)
            }
            Err(e) => {
                tracing::warn!(error = %e, "Real LLM unavailable; using stub LLM stage");
                Box::new(StubLlm)
            }
        };
        let tts: Box<dyn TtsStage> = match crate::rt_tts::KokoroTts::from_env() {
            Ok(model) => {
                tracing::info!("Realtime pipeline using real Kokoro TTS stage");
                Box::new(model)
            }
            Err(e) => {
                tracing::warn!(error = %e, "Real TTS unavailable; using stub TTS stage");
                Box::new(StubTts)
            }
        };
        Self { stt, llm, tts }
    }
}

/// Helper to create a standalone real STT stage based on `STT_BACKEND`.
/// Returns `None` if the model failed to load.
///
/// Prefer [`crate::stage_pools::StagePools::try_acquire_streaming_stt`] for
/// realtime sessions so model instances are shared across connections.
pub fn create_real_stt() -> Option<Box<dyn SttStage>> {
    use crate::stage_pools::{dev_model_selection, SttModel};
    match dev_model_selection() {
        SttModel::Zipformer => {
            crate::rt_stt_zipformer::SherpaZipformer::from_env().ok().map(|m| Box::new(m) as Box<dyn SttStage>)
        }
        SttModel::Streaming => {
            crate::rt_stt_eou::ParakeetEouStt::from_env().ok().map(|m| Box::new(m) as Box<dyn SttStage>)
        }
        SttModel::Parakeet => {
            crate::rt_stt::ParakeetStt::from_env().ok().map(|m| Box::new(m) as Box<dyn SttStage>)
        }
    }
}

// ===========================================================================
// Phase-1 stub stages (deterministic; NOT real models)
// ===========================================================================

/// STUB STT: reports how much audio it "heard". Replace with Parakeet TDT.
pub struct StubStt;

impl SttStage for StubStt {
    fn transcribe(&mut self, pcm: &[u8], partial: &mut dyn FnMut(&str)) -> String {
        let ms = pcm.len() / (PIPELINE_SAMPLE_RATE as usize * 2 / 1000).max(1);
        partial("[stub-stt] transcribing…");
        format!("[stub-stt transcript of {ms} ms of audio]")
    }
}

/// STUB LLM: echoes a canned reply, streamed word-by-word so the token-delta
/// path is exercised. Replace with the Qwen KV-cache loop.
pub struct StubLlm;

impl LlmStage for StubLlm {
    fn generate(
        &mut self,
        prompt: &str,
        _instructions: Option<&str>,
        cancel: &AtomicBool,
        on_token: &mut dyn FnMut(&str),
    ) -> String {
        let reply = format!("[stub-llm] I heard: {prompt}. This is a placeholder reply.");
        let mut out = String::new();
        for word in reply.split_inclusive(' ') {
            if cancel.load(Ordering::Relaxed) {
                break;
            }
            on_token(word);
            out.push_str(word);
        }
        out
    }
}

/// STUB TTS: emits silent PCM16 sized to the text length, in 20 ms chunks, so
/// the audio-delta streaming + barge-in cancellation path is exercised.
/// Replace with Kokoro + G2P.
pub struct StubTts;

impl TtsStage for StubTts {
    fn synthesize(
        &mut self,
        text: &str,
        _voice: Option<&str>,
        cancel: &AtomicBool,
        on_audio: &mut dyn FnMut(&[u8]),
    ) {
        // ~60 ms of audio per character, emitted as 20 ms (640-byte) chunks.
        const CHUNK_BYTES: usize = (PIPELINE_SAMPLE_RATE as usize) * 2 * 20 / 1000; // 640
        let total_chunks = (text.chars().count() * 3).max(1);
        let silence = vec![0u8; CHUNK_BYTES];
        for _ in 0..total_chunks {
            if cancel.load(Ordering::Relaxed) {
                break;
            }
            on_audio(&silence);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validate_pcm16_chunk_accepts_even_and_rejects_odd() {
        assert!(validate_pcm16_chunk(&[]).is_ok());
        assert!(validate_pcm16_chunk(&[0u8, 1u8]).is_ok());
        assert!(validate_pcm16_chunk(&[0u8]).is_err());
        let huge = vec![0u8; MAX_AUDIO_CHUNK_BYTES + 1];
        assert!(validate_pcm16_chunk(&huge).is_err());
    }

    #[test]
    fn stub_stt_reports_duration() {
        let mut stt = StubStt;
        let pcm = vec![0u8; 640 * 5]; // 100 ms
        let mut partials = 0;
        let final_text = stt.transcribe(&pcm, &mut |_| partials += 1);
        assert!(partials >= 1);
        assert!(final_text.contains("100 ms") || final_text.contains("ms"));
    }

    #[test]
    fn stub_llm_streams_tokens_and_respects_cancel() {
        let mut llm = StubLlm;
        // Uncancelled: streams multiple tokens.
        let cancel = AtomicBool::new(false);
        let mut tokens = 0;
        let full = llm.generate("hello", None, &cancel, &mut |_| tokens += 1);
        assert!(tokens > 1);
        assert!(full.contains("hello"));

        // Pre-cancelled: stops immediately.
        let cancelled = AtomicBool::new(true);
        let mut tokens2 = 0;
        let out = llm.generate("hello", None, &cancelled, &mut |_| tokens2 += 1);
        assert_eq!(tokens2, 0);
        assert!(out.is_empty());
    }

    #[test]
    fn stub_tts_emits_chunks_and_respects_cancel() {
        let mut tts = StubTts;
        let cancel = AtomicBool::new(false);
        let mut chunks = 0;
        tts.synthesize("hi", None, &cancel, &mut |c| {
            assert_eq!(c.len(), 640);
            chunks += 1;
        });
        assert!(chunks > 0);

        let cancelled = AtomicBool::new(true);
        let mut chunks2 = 0;
        tts.synthesize("hi", None, &cancelled, &mut |_| chunks2 += 1);
        assert_eq!(chunks2, 0);
    }

    #[test]
    fn pipeline_stubbed_constructs() {
        let mut p = Pipeline::stubbed();
        let cancel = AtomicBool::new(false);
        let transcript = p.stt.transcribe(&vec![0u8; 640], &mut |_| {});
        let mut reply = String::new();
        p.llm.generate(&transcript, None, &cancel, &mut |t| reply.push_str(t));
        let mut audio_len = 0;
        p.tts.synthesize(&reply, None, &cancel, &mut |c| audio_len += c.len());
        assert!(audio_len > 0);
    }
}
