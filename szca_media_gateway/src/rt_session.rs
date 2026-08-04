/// Realtime voice session loop: ties WebSocket transport, dialect translation,
/// server-side VAD turn detection, the STT->LLM->TTS pipeline, and barge-in
/// into one bidirectional streaming conversation.
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use axum::extract::ws::{Message, WebSocket};
use futures_util::{SinkExt, StreamExt};
use tokio::sync::mpsc;
use tokio::time::{timeout, Duration};

use crossbeam_channel as cb;

use crate::rt_events::{ClientCommand, ServerEvent, SessionSettings};
use crate::rt_pipeline::{LlmStage, SttStage, TtsStage, validate_pcm16_chunk};
use crate::rt_protocol::{Dialect, DialectKind};
use crate::stage_pools::{LlmPoolAdapter, StagePools, SttPoolAdapter, TtsPoolAdapter};
use crate::dsp::DspProcessor;
use crate::vad::{VadConfig, VadEvent, VadProcessor};

const EVENT_CHANNEL_CAPACITY: usize = 256;
const VAD_FRAME_BYTES: usize = 640;
/// Cap on buffered utterance PCM (30 s @ 16 kHz mono PCM16). Prevents unbounded
/// growth if a client streams without turn boundaries; StagePool backpressure
/// covers inference queue depth separately.
const MAX_UTTERANCE_BYTES: usize = 16_000 * 2 * 30;
/// Application-layer WebSocket ping interval. Keeps half-open connections from
/// sitting idle until the 35 s read timeout; clients reply with Pong (handled
/// automatically by tungstenite for inbound Pings).
const WS_PING_INTERVAL: Duration = Duration::from_secs(30);
const WS_IDLE_TIMEOUT: Duration = Duration::from_secs(35);
/// Pre-speech audio held for lookback when VAD triggers slightly after onset (~200 ms).
const LOOKBACK_BYTES: usize = 16_000 * 2 * 200 / 1000;

pub struct RealtimeConfig {
    pub dialect: DialectKind,
    pub vad: VadConfig,
    pub pools: Option<Arc<StagePools>>,
    pub dsp: Option<Box<DspProcessor>>,
}

#[derive(Clone, Default)]
pub struct Settings {
    pub instructions: Option<String>,
    pub voice: Option<String>,
    pub auto_turn: bool,
}

impl Settings {
    fn apply(&mut self, s: SessionSettings) {
        if s.instructions.is_some() { self.instructions = s.instructions; }
        if s.voice.is_some() { self.voice = s.voice; }
        if let Some(auto) = s.auto_turn_detection { self.auto_turn = auto; }
    }
}

/// Run a full realtime session over socket.
pub async fn run_session(socket: WebSocket, session_id: String, mut config: RealtimeConfig) {
    let (mut ws_sink, mut ws_stream) = socket.split();
    let (tx_evt, mut rx_evt) = mpsc::channel::<ServerEvent>(EVENT_CHANNEL_CAPACITY);
    let encoder: Box<dyn Dialect> = config.dialect.adapter();
    let pools = config.pools;
    let decoder: Box<dyn Dialect> = config.dialect.adapter();
    let mut vad = VadProcessor::new(config.vad);
    let mut settings = Settings { auto_turn: true, ..Default::default() };
    let mut utterance: Vec<u8> = Vec::new();
    let mut vad_carry: Vec<u8> = Vec::new();
    let mut lookback_ring: Vec<u8> = Vec::with_capacity(LOOKBACK_BYTES);
    let mut lookback_applied = false;
    let responding = Arc::new(AtomicBool::new(false));
    let cancel = Arc::new(AtomicBool::new(false));
    let mut streaming_stt: Option<crate::stage_pools::StreamingSttHandle> =
        pools.as_ref().and_then(|p| p.try_acquire_streaming_stt());
    if streaming_stt.is_some() {
        tracing::info!(session_id = %session_id, "Acquired shared streaming STT lease");
    }
    let streaming_supports_lookback = streaming_stt
        .as_ref()
        .map(|s| s.supports_lookback())
        .unwrap_or(false);
    let mut streaming_transcript = String::new();

    // Cancellation token for orphan task cleanup
    let (cancel_tx, _cancel_rx) = tokio::sync::broadcast::channel(1);
    let cancel_token = Arc::new(cancel_tx);

    // Outbound task: ServerEvents → dialect text, plus periodic WS pings so idle
    // sessions are still probed before WS_IDLE_TIMEOUT closes them.
    let oc = cancel_token.clone();
    let outbound = tokio::spawn(async move {
        let mut cancel_rx = oc.subscribe();
        let mut ping = tokio::time::interval(WS_PING_INTERVAL);
        ping.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
        // Skip the immediate first tick so we don't ping on connect.
        ping.tick().await;
        loop {
            tokio::select! {
                Some(event) = rx_evt.recv() => {
                    if let Some(text) = encoder.encode(&event) {
                        if ws_sink.send(Message::Text(text)).await.is_err() { break; }
                    }
                }
                _ = ping.tick() => {
                    if ws_sink.send(Message::Ping(Vec::new())).await.is_err() { break; }
                }
                _ = cancel_rx.recv() => { break; }
            }
        }
    });

    let _ = tx_evt.send(ServerEvent::SessionCreated {
        session_id: session_id.clone(),
    }).await;

    'session: loop {
        // Any inbound frame (incl. Pong for our Ping) resets the idle timer.
        match timeout(WS_IDLE_TIMEOUT, ws_stream.next()).await {
            Ok(Some(Ok(Message::Text(t)))) => {
                for cmd in decoder.decode(&t) {
                    match cmd {
                        ClientCommand::UpdateSession(s) => settings.apply(s),
                        ClientCommand::AppendAudio(pcm) => {
                            if let Err(e) = validate_pcm16_chunk(&pcm) {
                                tracing::warn!(error = %e, "Rejecting invalid audio chunk");
                                let _ = tx_evt.send(ServerEvent::Error { message: e }).await;
                                continue;
                            }
                            let cleaned = if let Some(ref mut dsp) = config.dsp {
                                match dsp.process(&pcm) {
                                    Ok(out) => out,
                                    Err(e) => { tracing::warn!(error = %e, "DSP failed"); pcm.clone() }
                                }
                            } else { pcm.clone() };
                            let vad_out = feed_vad(
                                &mut vad, &mut vad_carry, &cleaned,
                                &responding, &cancel, &tx_evt,
                            ).await;
                            if vad_out.speech_started
                                && streaming_supports_lookback
                                && !lookback_applied
                                && !lookback_ring.is_empty()
                            {
                                apply_speech_lookback(
                                    &lookback_ring,
                                    &mut utterance,
                                    streaming_stt.as_mut(),
                                    &mut streaming_transcript,
                                    &tx_evt,
                                ).await;
                                lookback_applied = true;
                            }
                            push_lookback_ring(&mut lookback_ring, &cleaned, LOOKBACK_BYTES);
                            append_utterance(&mut utterance, &cleaned);
                            let mut early_turn_ended = false;
                            if !responding.load(Ordering::Relaxed) {
                                if let Some(ref mut stt) = streaming_stt {
                                    if let Some(chunk_res) = stt.push_chunk(&cleaned) {
                                        if !chunk_res.delta_text.is_empty() {
                                            streaming_transcript.push_str(&chunk_res.delta_text);
                                            let _ = tx_evt.send(ServerEvent::TranscriptDelta { text: chunk_res.delta_text }).await;
                                        }
                                        if chunk_res.end_of_utterance {
                                            let _ = tx_evt.send(ServerEvent::SpeechStopped).await;
                                            early_turn_ended = true;
                                        }
                                    }
                                }
                            }
                            if (vad_out.turn_ended || early_turn_ended) && settings.auto_turn {
                                let pre = if streaming_stt.is_some() && !streaming_transcript.trim().is_empty() {
                                    Some(streaming_transcript.trim().to_string())
                                } else { None };
                                if let Some(ref mut stt) = streaming_stt { stt.reset_stream(); streaming_transcript.clear(); }
                                lookback_applied = false;
                                maybe_start_response(&pools, &mut utterance, pre, &settings, &responding, &cancel, &tx_evt);
                            }
                        }
                        ClientCommand::CommitAudio | ClientCommand::CreateResponse => {
                            let pre = if streaming_stt.is_some() && !streaming_transcript.trim().is_empty() {
                                Some(streaming_transcript.trim().to_string())
                            } else { None };
                            if let Some(ref mut stt) = streaming_stt { stt.reset_stream(); streaming_transcript.clear(); }
                            maybe_start_response(&pools, &mut utterance, pre, &settings, &responding, &cancel, &tx_evt);
                        }
                        ClientCommand::Cancel => {
                            if responding.load(Ordering::Relaxed) { cancel.store(true, Ordering::Relaxed); }
                            if let Some(ref mut stt) = streaming_stt { stt.reset_stream(); streaming_transcript.clear(); }
                            utterance.clear();
                            lookback_applied = false;
                        }
                        ClientCommand::Hangup => {
                            cancel.store(true, Ordering::Relaxed);
                            break 'session;
                        }
                    }
                }
            }
            // Close / error / EOF / idle timeout → end session.
            Ok(Some(Ok(Message::Close(_)))) | Ok(Some(Err(_))) | Ok(None) | Err(_) => break,
            // Binary / Ping / Pong: Ping is auto-answered by tungstenite; Pong
            // and ignored Binary still reset the idle timeout via this arm.
            Ok(Some(Ok(_))) => {}
        }
    }

    // Cleanup: cancel all tasks, wait for them
    cancel.store(true, Ordering::Relaxed);
    cancel_token.send(()).ok();
    drop(tx_evt);
    let _ = outbound.await;
    tracing::info!(session_id = %session_id, "Session ended");
}

/// Outcome of feeding one PCM chunk through VAD frame processing.
struct VadFeedOutcome {
    turn_ended: bool,
    speech_started: bool,
}

/// Maintain a rolling buffer of the most recent `cap` bytes of PCM.
fn push_lookback_ring(ring: &mut Vec<u8>, pcm: &[u8], cap: usize) {
    if pcm.is_empty() || cap == 0 {
        return;
    }
    ring.extend_from_slice(pcm);
    if ring.len() > cap {
        let drop_n = ring.len() - cap;
        ring.drain(..drop_n);
    }
}

/// Prepend pre-speech lookback to the utterance and streaming STT path.
async fn apply_speech_lookback(
    lookback: &[u8],
    utterance: &mut Vec<u8>,
    streaming_stt: Option<&mut crate::stage_pools::StreamingSttHandle>,
    streaming_transcript: &mut String,
    tx_evt: &mpsc::Sender<ServerEvent>,
) {
    let mut prefix = lookback.to_vec();
    prefix.extend_from_slice(utterance);
    *utterance = prefix;
    if let Some(stt) = streaming_stt {
        if let Some(chunk_res) = stt.push_chunk(lookback) {
            if !chunk_res.delta_text.is_empty() {
                streaming_transcript.push_str(&chunk_res.delta_text);
                let _ = tx_evt.send(ServerEvent::TranscriptDelta { text: chunk_res.delta_text }).await;
            }
        }
    }
}

/// Append PCM to the utterance buffer, dropping oldest bytes if over the cap.
fn append_utterance(utterance: &mut Vec<u8>, pcm: &[u8]) {
    let new_len = utterance.len().saturating_add(pcm.len());
    if new_len > MAX_UTTERANCE_BYTES {
        let overflow = new_len - MAX_UTTERANCE_BYTES;
        let drop_n = overflow.min(utterance.len());
        if drop_n > 0 {
            tracing::warn!(
                buffered = utterance.len(),
                drop_bytes = drop_n,
                "utterance buffer at cap; dropping oldest PCM"
            );
            utterance.drain(..drop_n);
        }
        // If a single chunk alone exceeds the cap, keep only its tail.
        if pcm.len() > MAX_UTTERANCE_BYTES {
            utterance.clear();
            utterance.extend_from_slice(&pcm[pcm.len() - MAX_UTTERANCE_BYTES..]);
            return;
        }
    }
    utterance.extend_from_slice(pcm);
}

/// Feed PCM to VAD in 20ms frames, emit speech start/stop events.
async fn feed_vad(
    vad: &mut VadProcessor,
    vad_carry: &mut Vec<u8>,
    pcm: &[u8],
    responding: &Arc<AtomicBool>,
    cancel: &Arc<AtomicBool>,
    tx_evt: &mpsc::Sender<ServerEvent>,
) -> VadFeedOutcome {
    let mut outcome = VadFeedOutcome {
        turn_ended: false,
        speech_started: false,
    };
    vad_carry.extend_from_slice(pcm);
    while vad_carry.len() >= VAD_FRAME_BYTES {
        let frame: Vec<u8> = vad_carry.drain(..VAD_FRAME_BYTES).collect();
        let tts_playing = responding.load(Ordering::Relaxed);
        match vad.process(&frame, tts_playing) {
            VadEvent::SpeechStart => {
                outcome.speech_started = true;
                let _ = tx_evt.send(ServerEvent::SpeechStarted).await;
            }
            VadEvent::BargeIn => {
                cancel.store(true, Ordering::Relaxed);
                outcome.speech_started = true;
                let _ = tx_evt.send(ServerEvent::SpeechStarted).await;
            }
            VadEvent::SpeechEnd => {
                let _ = tx_evt.send(ServerEvent::SpeechStopped).await;
                outcome.turn_ended = true;
            }
            VadEvent::Speech | VadEvent::Silence => {}
        }
    }
    outcome
}

/// Start a response turn if not already running and audio is available.
fn maybe_start_response(
    pools: &Option<Arc<StagePools>>,
    utterance: &mut Vec<u8>,
    pre_transcript: Option<String>,
    settings: &Settings,
    responding: &Arc<AtomicBool>,
    cancel: &Arc<AtomicBool>,
    tx_evt: &mpsc::Sender<ServerEvent>,
) {
    if utterance.is_empty() { return; }
    if responding.compare_exchange(false, true, Ordering::AcqRel, Ordering::Relaxed).is_err() { return; }
    cancel.store(false, Ordering::Relaxed);
    let pcm = std::mem::take(utterance);
    let pools = pools.as_ref().map(Arc::clone);
    let responding = Arc::clone(responding);
    let cancel = Arc::clone(cancel);
    let tx = tx_evt.clone();
    let settings = settings.clone();
    tokio::task::spawn_blocking(move || {
        run_response(pools, pcm, pre_transcript, settings, cancel, responding, tx);
    });
}

/// Blocking STT -> LLM -> TTS worker for one turn.
fn run_response(
    pools: Option<Arc<StagePools>>,
    pcm: Vec<u8>,
    pre_transcript: Option<String>,
    settings: Settings,
    cancel: Arc<AtomicBool>,
    responding: Arc<AtomicBool>,
    tx: mpsc::Sender<ServerEvent>,
) {
    let Some(pools) = pools else {
        tracing::warn!("No inference pools; skipping response");
        let _ = tx.blocking_send(ServerEvent::Error { message: "inference not available".into() });
        responding.store(false, Ordering::Relaxed);
        return;
    };
    let mut stt_adapter = SttPoolAdapter::from_pools(&pools);
    let mut llm_adapter = pools.llm.as_ref().map(LlmPoolAdapter::new);
    let mut tts_adapter = pools.tts.as_ref().map(TtsPoolAdapter::new);
    run_turn(
        TurnStages {
            stt: stt_adapter.as_mut().map(|s| s as &mut dyn SttStage),
            llm: llm_adapter.as_mut().map(|l| l as &mut dyn LlmStage),
            tts: tts_adapter.as_mut().map(|t| t as &mut dyn TtsStage),
        },
        pcm, pre_transcript, settings, cancel, responding, tx,
    );
}

/// The three (optional) stages a single turn drives.
pub struct TurnStages<'a> {
    pub stt: Option<&'a mut dyn SttStage>,
    pub llm: Option<&'a mut dyn LlmStage>,
    pub tts: Option<&'a mut dyn TtsStage>,
}

/// Events sent from LLM thread to calling thread during parallel LLM || TTS.
enum LlmEvent {
    TextDelta(String),
    Sentence(String),
}

/// STT -> LLM -> TTS turn: the single implementation of the per-turn event contract.
pub fn run_turn(
    stages: TurnStages<'_>,
    pcm: Vec<u8>,
    pre_transcript: Option<String>,
    settings: Settings,
    cancel: Arc<AtomicBool>,
    responding: Arc<AtomicBool>,
    tx: mpsc::Sender<ServerEvent>,
) {
    let TurnStages { stt: mut stt_adapter, llm: mut llm_adapter, tts: mut tts_adapter } = stages;

    // ---- STT ----
    let transcript;
    if let Some(pre) = pre_transcript {
        transcript = pre;
    } else if let Some(ref mut stt) = stt_adapter {
        let mut partial_cb = |t: &str| { let _ = tx.blocking_send(ServerEvent::TranscriptDelta { text: t.to_string() }); };
        transcript = stt.transcribe(&pcm, &mut partial_cb);
    } else {
        tracing::warn!("STT not available; empty transcript");
        transcript = String::new();
    }
    let _ = tx.blocking_send(ServerEvent::TranscriptDone { text: transcript.clone() });

    if cancel.load(Ordering::Relaxed) { return finish(&tx, &responding, true); }

    // ---- LLM || TTS (parallel, overlapping) ----
    let voice = settings.voice.clone();
    let (event_tx, event_rx) = cb::unbounded::<LlmEvent>();
    let (done_tx, done_rx) = cb::bounded::<String>(1);

    std::thread::scope(|s| {
        // LLM thread: generate tokens, push text/sentences to channel
        s.spawn(|| {
            let mut reply = String::new();
            let mut tts_buf = String::new();
            let mut on_token = |t: &str| {
                reply.push_str(t);
                let _ = event_tx.send(LlmEvent::TextDelta(t.to_string()));
                tts_buf.push_str(t);
                while let Some(sentence) = take_ready_sentence(&mut tts_buf) {
                    if cancel.load(Ordering::Relaxed) { break; }
                    if event_tx.send(LlmEvent::Sentence(sentence)).is_err() { break; }
                }
            };
            if let Some(ref mut llm) = llm_adapter {
                llm.generate(&transcript, settings.instructions.as_deref(), &cancel, &mut on_token);
            }
            let tail = tts_buf.trim().to_string();
            if !tail.is_empty() && !cancel.load(Ordering::Relaxed) {
                let _ = event_tx.send(LlmEvent::Sentence(tail));
            }
            drop(event_tx);
            let _ = done_tx.send(reply);
        });

        // Calling thread: read events, synthesize TTS, forward to client
        while let Ok(event) = event_rx.recv() {
            match event {
                LlmEvent::TextDelta(text) => { let _ = tx.blocking_send(ServerEvent::TextDelta { text }); }
                LlmEvent::Sentence(text) => {
                    if let Some(ref mut tts) = tts_adapter {
                        let mut on_audio = |chunk: &[u8]| { let _ = tx.blocking_send(ServerEvent::AudioDelta { pcm: chunk.to_vec() }); };
                        tts.synthesize(&text, voice.as_deref(), &cancel, &mut on_audio);
                    }
                }
            }
            if cancel.load(Ordering::Relaxed) { break; }
        }
    });

    let reply = done_rx.recv().unwrap_or_default();
    let _ = tx.blocking_send(ServerEvent::TextDone { text: reply });
    let _ = tx.blocking_send(ServerEvent::AudioDone);
    finish(&tx, &responding, cancel.load(Ordering::Relaxed));
}

/// Extract the leading complete sentence from buf, if one is ready.
fn take_ready_sentence(buf: &mut String) -> Option<String> {
    let end = buf.char_indices()
        .find(|(_, c)| matches!(c, '.' | '!' | '?' | '\n'))
        .map(|(i, c)| i + c.len_utf8())?;
    let sentence: String = buf[..end].trim().to_string();
    buf.drain(..end);
    if !sentence.chars().any(char::is_alphanumeric) { return take_ready_sentence(buf); }
    Some(sentence)
}

/// Emit the terminal event and release the responding slot.
fn finish(tx: &mpsc::Sender<ServerEvent>, responding: &Arc<AtomicBool>, cancelled: bool) {
    if cancelled { let _ = tx.blocking_send(ServerEvent::ResponseCancelled); }
    else { let _ = tx.blocking_send(ServerEvent::ResponseDone); }
    responding.store(false, Ordering::Relaxed);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sentence_splitter_yields_complete_sentences() {
        let mut buf = String::from("Hello there. How are");
        assert_eq!(take_ready_sentence(&mut buf).as_deref(), Some("Hello there."));
        assert_eq!(take_ready_sentence(&mut buf), None);
        assert_eq!(buf, " How are");
        buf.push_str(" you?");
        assert_eq!(take_ready_sentence(&mut buf).as_deref(), Some("How are you?"));
        assert_eq!(take_ready_sentence(&mut buf), None);
    }

    #[test]
    fn sentence_splitter_handles_all_terminators() {
        let mut buf = String::from("One! Two? Three\nfour");
        assert_eq!(take_ready_sentence(&mut buf).as_deref(), Some("One!"));
        assert_eq!(take_ready_sentence(&mut buf).as_deref(), Some("Two?"));
        assert_eq!(take_ready_sentence(&mut buf).as_deref(), Some("Three"));
        assert_eq!(take_ready_sentence(&mut buf), None);
        assert_eq!(buf, "four");
    }

    #[test]
    fn sentence_splitter_skips_empty_content() {
        let mut only = String::from("...done.");
        assert_eq!(take_ready_sentence(&mut only).as_deref(), Some("done."));
    }

    #[test]
    fn append_utterance_respects_byte_cap() {
        let mut buf = Vec::new();
        // Fill to just under the cap.
        append_utterance(&mut buf, &vec![1u8; MAX_UTTERANCE_BYTES - 10]);
        assert_eq!(buf.len(), MAX_UTTERANCE_BYTES - 10);
        // Overflow by 20 → drop 10 oldest, end at cap with the new tail.
        append_utterance(&mut buf, &vec![2u8; 20]);
        assert_eq!(buf.len(), MAX_UTTERANCE_BYTES);
        assert!(buf.ends_with(&[2u8; 20]));
    }

    #[test]
    fn append_utterance_single_chunk_over_cap_keeps_tail() {
        let mut buf = Vec::new();
        let mut big = vec![0u8; MAX_UTTERANCE_BYTES + 100];
        let last = big.len() - 1;
        big[last] = 0xAB;
        append_utterance(&mut buf, &big);
        assert_eq!(buf.len(), MAX_UTTERANCE_BYTES);
        assert_eq!(*buf.last().unwrap(), 0xAB);
    }

    #[test]
    fn lookback_ring_caps_at_max_bytes() {
        let mut ring = Vec::new();
        let cap = 100;
        push_lookback_ring(&mut ring, &vec![1u8; 60], cap);
        assert_eq!(ring.len(), 60);
        push_lookback_ring(&mut ring, &vec![2u8; 60], cap);
        assert_eq!(ring.len(), cap);
        assert_eq!(ring[cap - 1], 2);
    }
}
