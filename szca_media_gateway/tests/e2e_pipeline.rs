//! End-to-end realtime turn integration test: PCM -> STT -> LLM -> TTS -> PCM.
//!
//! This exercises the REAL production turn path ([`rt_session::run_turn`]) —
//! the same function the WebSocket session's `spawn_blocking` worker calls —
//! driven with the deterministic Phase-1 stub stages. No model weights and no
//! network are required, so it runs everywhere (including CI).
//!
//! What is asserted:
//!   * the full event ORDER contract per turn:
//!       TranscriptDelta* -> TranscriptDone -> (TextDelta | AudioDelta)*
//!       -> TextDone -> AudioDone -> ResponseDone
//!   * audio actually flows back as PCM16 frames (byte-aligned, non-empty);
//!   * TTS is interleaved with LLM generation (first AudioDelta arrives BEFORE
//!     TextDone), i.e. sentence-chunked streaming really is streaming;
//!   * barge-in: a cancel set before the turn starts yields ResponseCancelled
//!     and no audio;
//!   * mid-generation cancel stops the turn early and still releases the
//!     `responding` slot;
//!   * both wire dialects encode the whole event stream without dropping the
//!     terminal event (OpenAI Realtime + Gemini Live round-trip);
//!   * the VAD turn-detector drives an utterance boundary from raw PCM
//!     (speech -> silence -> SpeechEnd), which is what triggers a turn.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use szca_media_gateway::rt_events::ServerEvent;
use szca_media_gateway::rt_pipeline::{
    LlmStage, Pipeline, SttStage, TtsStage, PIPELINE_SAMPLE_RATE,
};
use szca_media_gateway::rt_protocol::{Dialect, DialectKind};
use szca_media_gateway::rt_session::{run_turn, Settings, TurnStages};
use szca_media_gateway::vad::{VadConfig, VadEvent, VadProcessor};
use tokio::sync::mpsc;

/// Bytes in one 20 ms PCM16 mono 16 kHz frame.
const FRAME_BYTES: usize = (PIPELINE_SAMPLE_RATE as usize) * 2 * 20 / 1000;

/// Build `ms` of loud voiced-like PCM16 (LE) so the RMS VAD fallback sees speech.
fn speech_pcm(ms: usize) -> Vec<u8> {
    let samples = PIPELINE_SAMPLE_RATE as usize * ms / 1000;
    let mut out = Vec::with_capacity(samples * 2);
    for i in 0..samples {
        let t = i as f64 / PIPELINE_SAMPLE_RATE as f64;
        let v = 0.6 * (2.0 * std::f64::consts::PI * 150.0 * t).sin()
            + 0.3 * (2.0 * std::f64::consts::PI * 450.0 * t).sin();
        out.extend_from_slice(&((v * 14000.0) as i16).to_le_bytes());
    }
    out
}

/// Build `ms` of digital silence.
fn silence_pcm(ms: usize) -> Vec<u8> {
    vec![0u8; PIPELINE_SAMPLE_RATE as usize * 2 * ms / 1000]
}

/// Short label for a `ServerEvent`, so ordering can be asserted on a Vec<&str>.
fn tag(e: &ServerEvent) -> &'static str {
    match e {
        ServerEvent::SessionCreated { .. } => "session_created",
        ServerEvent::SpeechStarted => "speech_started",
        ServerEvent::SpeechStopped => "speech_stopped",
        ServerEvent::TranscriptDelta { .. } => "transcript_delta",
        ServerEvent::TranscriptDone { .. } => "transcript_done",
        ServerEvent::TextDelta { .. } => "text_delta",
        ServerEvent::TextDone { .. } => "text_done",
        ServerEvent::AudioDelta { .. } => "audio_delta",
        ServerEvent::AudioDone => "audio_done",
        ServerEvent::ResponseCancelled => "response_cancelled",
        ServerEvent::ResponseDone => "response_done",
        ServerEvent::Error { .. } => "error",
    }
}

/// Drive one full turn through the production `run_turn` with stub stages and
/// collect every emitted event in order.
///
/// `pre_cancel` sets the shared cancel flag before the turn begins (barge-in
/// arriving during the user's own speech). Returns `(events, responding_flag)`.
async fn drive_turn(pcm: Vec<u8>, pre_cancel: bool) -> (Vec<ServerEvent>, Arc<AtomicBool>) {
    // Generous channel so the blocking worker never stalls on a full queue.
    let (tx, mut rx) = mpsc::channel::<ServerEvent>(4096);
    let cancel = Arc::new(AtomicBool::new(pre_cancel));
    let responding = Arc::new(AtomicBool::new(true));

    let cancel_w = Arc::clone(&cancel);
    let responding_w = Arc::clone(&responding);
    let worker = tokio::task::spawn_blocking(move || {
        // `Pipeline::stubbed()` owns boxed stub stages; borrow them as the three
        // trait objects `run_turn` expects.
        let mut pipeline = Pipeline::stubbed();
        run_turn(
            TurnStages {
                stt: Some(&mut *pipeline.stt as &mut dyn SttStage),
                llm: Some(&mut *pipeline.llm as &mut dyn LlmStage),
                tts: Some(&mut *pipeline.tts as &mut dyn TtsStage),
            },
            pcm,
            None,
            Settings {
                instructions: Some("Be brief.".into()),
                voice: Some("af_heart".into()),
                auto_turn: true,
            },
            cancel_w,
            responding_w,
            tx,
        );
    });

    let mut events = Vec::new();
    while let Some(e) = rx.recv().await {
        events.push(e);
    }
    worker.await.expect("turn worker should not panic");
    (events, responding)
}

#[tokio::test]
async fn full_turn_streams_transcript_text_and_audio_in_order() {
    let (events, responding) = drive_turn(speech_pcm(500), false).await;
    let tags: Vec<&str> = events.iter().map(tag).collect();

    // No errors, and the turn completed rather than being cancelled.
    assert!(
        !tags.contains(&"error"),
        "unexpected error event in {tags:?}"
    );
    assert_eq!(
        tags.last(),
        Some(&"response_done"),
        "turn must end with response_done: {tags:?}"
    );
    assert!(
        !tags.contains(&"response_cancelled"),
        "uncancelled turn must not emit response_cancelled: {tags:?}"
    );

    // Every stage produced output.
    assert!(
        tags.contains(&"transcript_delta"),
        "STT must stream at least one partial: {tags:?}"
    );
    assert!(tags.contains(&"transcript_done"), "{tags:?}");
    assert!(
        tags.iter().filter(|t| **t == "text_delta").count() > 1,
        "LLM must stream multiple token deltas: {tags:?}"
    );
    assert!(tags.contains(&"text_done"), "{tags:?}");
    assert!(tags.contains(&"audio_delta"), "{tags:?}");
    assert!(tags.contains(&"audio_done"), "{tags:?}");

    // ---- Ordering contract ----
    let idx = |t: &str| tags.iter().position(|x| *x == t).expect("event present");
    let last = |t: &str| tags.iter().rposition(|x| *x == t).expect("event present");

    // STT fully precedes LLM.
    assert!(
        last("transcript_delta") < idx("transcript_done"),
        "transcript deltas must precede transcript_done: {tags:?}"
    );
    assert!(
        idx("transcript_done") < idx("text_delta"),
        "transcript_done must precede the first token: {tags:?}"
    );
    // LLM text finishes before audio is closed out, and audio closes before the
    // terminal event.
    assert!(
        last("text_delta") < idx("text_done"),
        "token deltas must precede text_done: {tags:?}"
    );
    assert!(
        last("audio_delta") < idx("audio_done"),
        "audio deltas must precede audio_done: {tags:?}"
    );
    assert!(
        idx("audio_done") < idx("response_done"),
        "audio_done must precede response_done: {tags:?}"
    );

    // ---- Interleaving: TTS starts before generation ends ----
    // This is the property that keeps time-to-first-audio low; if TTS were run
    // only after the full reply was generated, first audio would land after
    // text_done.
    assert!(
        idx("audio_delta") < idx("text_done"),
        "first audio must arrive before generation completes (sentence-chunked \
         interleaving): {tags:?}"
    );

    // ---- Payload sanity: real PCM16 frames come back ----
    let audio: Vec<&Vec<u8>> = events
        .iter()
        .filter_map(|e| match e {
            ServerEvent::AudioDelta { pcm } => Some(pcm),
            _ => None,
        })
        .collect();
    let total: usize = audio.iter().map(|p| p.len()).sum();
    assert!(total > 0, "no PCM bytes returned");
    assert_eq!(total % 2, 0, "PCM16 output must be sample-aligned");
    for chunk in &audio {
        assert!(!chunk.is_empty(), "empty audio chunk emitted");
        assert_eq!(chunk.len() % 2, 0, "audio chunk not sample-aligned");
    }

    // Transcript and reply text are non-empty and the reply echoes the transcript
    // (the stub LLM's contract), proving the STT->LLM handoff carried data.
    let transcript = events
        .iter()
        .find_map(|e| match e {
            ServerEvent::TranscriptDone { text } => Some(text.clone()),
            _ => None,
        })
        .expect("transcript_done present");
    let reply = events
        .iter()
        .find_map(|e| match e {
            ServerEvent::TextDone { text } => Some(text.clone()),
            _ => None,
        })
        .expect("text_done present");
    assert!(!transcript.trim().is_empty(), "empty transcript");
    assert!(!reply.trim().is_empty(), "empty reply");
    assert!(
        reply.contains(&transcript),
        "reply {reply:?} should carry the transcript {transcript:?} through the \
         STT->LLM handoff"
    );

    // The turn released its admission slot.
    assert!(
        !responding.load(Ordering::Relaxed),
        "responding flag must be cleared when the turn ends"
    );

    // The concatenated text deltas must reconstruct the final text exactly —
    // clients render from deltas, so a mismatch is a visible bug.
    let streamed: String = events
        .iter()
        .filter_map(|e| match e {
            ServerEvent::TextDelta { text } => Some(text.as_str()),
            _ => None,
        })
        .collect();
    assert_eq!(streamed, reply, "text deltas must reconstruct text_done");
}

#[tokio::test]
async fn barge_in_before_turn_starts_cancels_without_audio() {
    let (events, responding) = drive_turn(speech_pcm(500), true).await;
    let tags: Vec<&str> = events.iter().map(tag).collect();

    assert_eq!(
        tags.last(),
        Some(&"response_cancelled"),
        "pre-cancelled turn must end with response_cancelled: {tags:?}"
    );
    assert!(
        !tags.contains(&"response_done"),
        "cancelled turn must not also report done: {tags:?}"
    );
    assert!(
        !tags.contains(&"audio_delta"),
        "cancelled-before-generation turn must emit no audio: {tags:?}"
    );
    // STT already ran to completion before the cancel check, so the transcript
    // is still delivered — the contract is "no assistant output", not "nothing".
    assert!(tags.contains(&"transcript_done"), "{tags:?}");
    assert!(
        !responding.load(Ordering::Relaxed),
        "responding flag must be cleared on cancel"
    );
}

/// LLM stage that trips a shared cancel flag after `n` tokens, then delegates.
///
/// Barge-in is raced by nature (the VAD fires on the inbound task while the
/// worker generates), so a test that flipped the flag from the reader task would
/// be flaky: the blocking worker can fill the event channel's buffer before the
/// reader observes token 1, and then "cancel mid-generation" never actually
/// happens mid-generation. Tripping the flag from inside the generation callback
/// makes the cancellation point exact and the assertions deterministic.
struct CancelAfter<'a> {
    inner: &'a mut dyn LlmStage,
    cancel: Arc<AtomicBool>,
    after: usize,
}

impl LlmStage for CancelAfter<'_> {
    fn generate(
        &mut self,
        prompt: &str,
        instructions: Option<&str>,
        cancel: &AtomicBool,
        on_token: &mut dyn FnMut(&str),
    ) -> String {
        let mut seen = 0usize;
        let flag = Arc::clone(&self.cancel);
        let after = self.after;
        self.inner.generate(prompt, instructions, cancel, &mut |t| {
            on_token(t);
            seen += 1;
            if seen >= after {
                flag.store(true, Ordering::Relaxed);
            }
        })
    }
}

#[tokio::test]
async fn cancel_mid_generation_stops_turn_early() {
    let (tx, mut rx) = mpsc::channel::<ServerEvent>(4096);
    let cancel = Arc::new(AtomicBool::new(false));
    let responding = Arc::new(AtomicBool::new(true));

    let cancel_w = Arc::clone(&cancel);
    let responding_w = Arc::clone(&responding);
    let worker = tokio::task::spawn_blocking(move || {
        let mut pipeline = Pipeline::stubbed();
        // Barge-in lands on the very first assistant token.
        let mut llm = CancelAfter {
            inner: &mut *pipeline.llm,
            cancel: Arc::clone(&cancel_w),
            after: 1,
        };
        run_turn(
            TurnStages {
                stt: Some(&mut *pipeline.stt as &mut dyn SttStage),
                llm: Some(&mut llm as &mut dyn LlmStage),
                tts: Some(&mut *pipeline.tts as &mut dyn TtsStage),
            },
            speech_pcm(500),
            None,
            Settings {
                auto_turn: true,
                ..Default::default()
            },
            cancel_w,
            responding_w,
            tx,
        );
    });

    let mut events = Vec::new();
    while let Some(e) = rx.recv().await {
        events.push(e);
    }
    worker.await.expect("turn worker should not panic");

    let tags: Vec<&str> = events.iter().map(tag).collect();
    assert_eq!(
        tags.last(),
        Some(&"response_cancelled"),
        "mid-generation cancel must report response_cancelled: {tags:?}"
    );
    assert!(
        !tags.contains(&"response_done"),
        "cancelled turn must not also report done: {tags:?}"
    );
    assert!(
        !responding.load(Ordering::Relaxed),
        "responding flag must be cleared on mid-turn cancel"
    );

    // Exactly one token escaped before the flag was seen, and the turn is
    // strictly shorter than an uncancelled one.
    let cut_deltas = tags.iter().filter(|t| **t == "text_delta").count();
    assert_eq!(
        cut_deltas, 1,
        "cancel on token 1 must stop generation immediately: {tags:?}"
    );
    let (full, _) = drive_turn(speech_pcm(500), false).await;
    let full_deltas = full.iter().filter(|e| tag(e) == "text_delta").count();
    assert!(
        cut_deltas < full_deltas,
        "cancelled turn produced {cut_deltas} deltas, uncancelled {full_deltas} \
         — cancel did not shorten generation"
    );
}

#[tokio::test]
async fn both_dialects_encode_the_whole_turn() {
    let (events, _) = drive_turn(speech_pcm(500), false).await;

    for kind in [DialectKind::OpenAiRealtime, DialectKind::GeminiLive] {
        let enc: Box<dyn Dialect> = kind.adapter();
        let mut encoded = 0usize;
        let mut saw_terminal = false;
        for e in &events {
            if let Some(text) = enc.encode(e) {
                // Every emitted frame must be valid JSON on the wire.
                let v: serde_json::Value =
                    serde_json::from_str(&text).expect("dialect must emit valid JSON");
                assert!(v.is_object(), "dialect frame must be a JSON object: {text}");
                encoded += 1;
                if matches!(e, ServerEvent::ResponseDone) {
                    saw_terminal = true;
                }
            }
        }
        assert!(encoded > 0, "{kind:?} encoded nothing");
        assert!(
            saw_terminal,
            "{kind:?} must encode the terminal ResponseDone event"
        );
    }
}

#[tokio::test]
async fn vad_detects_the_utterance_boundary_that_triggers_a_turn() {
    // RMS fallback (no Silero weights) — deterministic and CI-safe.
    let mut vad = VadProcessor::new(VadConfig::default());

    let mut saw_start = false;
    let mut saw_end = false;

    let feed = |vad: &mut VadProcessor, pcm: &[u8], saw_start: &mut bool, saw_end: &mut bool| {
        for frame in pcm.chunks(FRAME_BYTES) {
            if frame.len() < FRAME_BYTES {
                break;
            }
            match vad.process(frame, false) {
                VadEvent::SpeechStart => *saw_start = true,
                VadEvent::SpeechEnd => *saw_end = true,
                _ => {}
            }
        }
    };

    // 400 ms of speech, then 700 ms of silence (> silence_duration_ms 500).
    feed(&mut vad, &speech_pcm(400), &mut saw_start, &mut saw_end);
    assert!(saw_start, "VAD must report SpeechStart on loud speech");
    assert!(!saw_end, "SpeechEnd must not fire while speech is ongoing");

    feed(&mut vad, &silence_pcm(700), &mut saw_start, &mut saw_end);
    assert!(
        saw_end,
        "VAD must report SpeechEnd after sustained silence (this is what starts \
         a response turn)"
    );
    assert!(!vad.is_speech_active(), "speech must no longer be active");
}
