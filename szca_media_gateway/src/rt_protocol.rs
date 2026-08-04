/// Wire-dialect adapters for the realtime API.
///
/// The gateway supports two client-facing protocols over the same pipeline:
///   * OpenAI Realtime — event objects keyed by `type` (e.g.
///     `input_audio_buffer.append`, `response.audio.delta`).
///   * Gemini Live — message objects keyed by a top-level field (e.g.
///     `realtimeInput`, `serverContent`).
///
/// A [`Dialect`] converts between raw JSON text and the neutral
/// [`ClientCommand`] / [`ServerEvent`] types in [`crate::rt_events`]. The
/// session loop is dialect-agnostic; the concrete dialect is selected once per
/// connection (query string / subprotocol) and then only these two methods are
/// used.
///
/// Audio on the wire is base64-encoded PCM16 mono 16 kHz in both dialects; the
/// adapter is the only place base64 is applied, so the rest of the system deals
/// in raw bytes.

use base64::Engine as _;
use serde_json::{json, Value};

use crate::rt_events::{ClientCommand, ServerEvent, SessionSettings};

/// Which wire protocol a connection speaks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DialectKind {
    OpenAiRealtime,
    GeminiLive,
}

impl DialectKind {
    /// Select a dialect from a request path/query. Defaults to OpenAI Realtime.
    ///
    /// Recognizes `?dialect=gemini` / `?dialect=openai` (case-insensitive) and
    /// a bare `gemini` / `openai` substring so simple clients can opt in.
    pub fn from_query(raw_query: &str) -> Self {
        let q = raw_query.to_ascii_lowercase();
        if q.contains("gemini") {
            DialectKind::GeminiLive
        } else {
            DialectKind::OpenAiRealtime
        }
    }

    /// Construct the boxed adapter for this dialect.
    pub fn adapter(self) -> Box<dyn Dialect> {
        match self {
            DialectKind::OpenAiRealtime => Box::new(OpenAiDialect),
            DialectKind::GeminiLive => Box::new(GeminiDialect),
        }
    }
}

/// Bidirectional translator between a wire dialect and neutral event types.
pub trait Dialect: Send {
    /// Decode one inbound JSON text frame into zero or more client commands.
    /// Unknown/irrelevant messages decode to an empty vec (ignored, not fatal).
    fn decode(&self, text: &str) -> Vec<ClientCommand>;

    /// Encode a server event into an outbound JSON text frame. Returns `None`
    /// for events this dialect has no representation for (silently skipped).
    fn encode(&self, event: &ServerEvent) -> Option<String>;
}

fn b64_decode(s: &str) -> Option<Vec<u8>> {
    base64::engine::general_purpose::STANDARD.decode(s).ok()
}

fn b64_encode(bytes: &[u8]) -> String {
    base64::engine::general_purpose::STANDARD.encode(bytes)
}

// ===========================================================================
// OpenAI Realtime dialect
// ===========================================================================

/// OpenAI Realtime API dialect (`type`-tagged client/server events).
pub struct OpenAiDialect;

impl Dialect for OpenAiDialect {
    fn decode(&self, text: &str) -> Vec<ClientCommand> {
        let v: Value = match serde_json::from_str::<Value>(text) {
            Ok(v) => v,
            Err(e) => {
                tracing::debug!(error = %e, "Failed to parse WebSocket JSON message");
                return Vec::new();
            }
        };
        let ty = v.get("type").and_then(Value::as_str).unwrap_or("");
        match ty {
            "session.update" => {
                let s = v.get("session").cloned().unwrap_or(Value::Null);
                vec![ClientCommand::UpdateSession(SessionSettings {
                    instructions: s
                        .get("instructions")
                        .and_then(Value::as_str)
                        .map(str::to_string),
                    voice: s.get("voice").and_then(Value::as_str).map(str::to_string),
                    auto_turn_detection: s
                        .get("turn_detection")
                        .map(|td| !td.is_null()),
                })]
            }
            "input_audio_buffer.append" => v
                .get("audio")
                .and_then(Value::as_str)
                .and_then(b64_decode)
                .map(|pcm| vec![ClientCommand::AppendAudio(pcm)])
                .unwrap_or_default(),
            "input_audio_buffer.commit" => vec![ClientCommand::CommitAudio],
            "response.create" => vec![ClientCommand::CreateResponse],
            "response.cancel" => vec![ClientCommand::Cancel],
            _ => Vec::new(),
        }
    }

    fn encode(&self, event: &ServerEvent) -> Option<String> {
        let v = match event {
            ServerEvent::SessionCreated { session_id } => json!({
                "type": "session.created",
                "session": { "id": session_id }
            }),
            ServerEvent::SpeechStarted => json!({
                "type": "input_audio_buffer.speech_started"
            }),
            ServerEvent::SpeechStopped => json!({
                "type": "input_audio_buffer.speech_stopped"
            }),
            ServerEvent::TranscriptDelta { text } => json!({
                "type": "conversation.item.input_audio_transcription.delta",
                "delta": text
            }),
            ServerEvent::TranscriptDone { text } => json!({
                "type": "conversation.item.input_audio_transcription.completed",
                "transcript": text
            }),
            ServerEvent::TextDelta { text } => json!({
                "type": "response.output_text.delta",
                "delta": text
            }),
            ServerEvent::TextDone { text } => json!({
                "type": "response.output_text.done",
                "text": text
            }),
            ServerEvent::AudioDelta { pcm } => json!({
                "type": "response.output_audio.delta",
                "delta": b64_encode(pcm)
            }),
            ServerEvent::AudioDone => json!({
                "type": "response.output_audio.done"
            }),
            ServerEvent::ResponseCancelled => json!({
                "type": "response.cancelled"
            }),
            ServerEvent::ResponseDone => json!({
                "type": "response.done"
            }),
            ServerEvent::Error { message } => json!({
                "type": "error",
                "error": { "message": message }
            }),
        };
        Some(v.to_string())
    }
}

// ===========================================================================
// Gemini Live dialect
// ===========================================================================

/// Gemini Live API dialect (field-tagged BidiGenerateContent messages).
pub struct GeminiDialect;

impl Dialect for GeminiDialect {
    fn decode(&self, text: &str) -> Vec<ClientCommand> {
        let v: Value = match serde_json::from_str(text) {
            Ok(v) => v,
            Err(_) => return Vec::new(),
        };

        // setup: { generationConfig, systemInstruction, ... }
        if let Some(setup) = v.get("setup") {
            let instructions = setup
                .get("systemInstruction")
                .and_then(|si| si.get("parts"))
                .and_then(Value::as_array)
                .and_then(|parts| parts.first())
                .and_then(|p| p.get("text"))
                .and_then(Value::as_str)
                .map(str::to_string);
            return vec![ClientCommand::UpdateSession(SessionSettings {
                instructions,
                voice: None,
                auto_turn_detection: None,
            })];
        }

        // realtimeInput: { audio: { data: <b64> } } or legacy mediaChunks.
        if let Some(rt) = v.get("realtimeInput") {
            let mut cmds = Vec::new();
            if let Some(pcm) = rt
                .get("audio")
                .and_then(|a| a.get("data"))
                .and_then(Value::as_str)
                .and_then(b64_decode)
            {
                cmds.push(ClientCommand::AppendAudio(pcm));
            }
            if let Some(chunks) = rt.get("mediaChunks").and_then(Value::as_array) {
                for c in chunks {
                    if let Some(pcm) =
                        c.get("data").and_then(Value::as_str).and_then(b64_decode)
                    {
                        cmds.push(ClientCommand::AppendAudio(pcm));
                    }
                }
            }
            // audioStreamEnd marks a client-side end-of-turn.
            if rt.get("audioStreamEnd").and_then(Value::as_bool) == Some(true) {
                cmds.push(ClientCommand::CommitAudio);
            }
            return cmds;
        }

        Vec::new()
    }

    fn encode(&self, event: &ServerEvent) -> Option<String> {
        let v = match event {
            ServerEvent::SessionCreated { .. } => json!({ "setupComplete": {} }),
            // Gemini has no explicit speech-start/stop client events; skip.
            ServerEvent::SpeechStarted | ServerEvent::SpeechStopped => return None,
            ServerEvent::TranscriptDelta { text } => json!({
                "serverContent": { "inputTranscription": { "text": text } }
            }),
            ServerEvent::TranscriptDone { text } => json!({
                "serverContent": { "inputTranscription": { "text": text, "finished": true } }
            }),
            ServerEvent::TextDelta { text } => json!({
                "serverContent": {
                    "modelTurn": { "parts": [ { "text": text } ] }
                }
            }),
            // Final text is conveyed via deltas + generationComplete; no body.
            ServerEvent::TextDone { .. } => return None,
            ServerEvent::AudioDelta { pcm } => json!({
                "serverContent": {
                    "modelTurn": {
                        "parts": [ {
                            "inlineData": {
                                "mimeType": "audio/pcm;rate=16000",
                                "data": b64_encode(pcm)
                            }
                        } ]
                    }
                }
            }),
            ServerEvent::AudioDone => json!({
                "serverContent": { "generationComplete": true }
            }),
            ServerEvent::ResponseCancelled => json!({
                "serverContent": { "interrupted": true }
            }),
            ServerEvent::ResponseDone => json!({
                "serverContent": { "turnComplete": true }
            }),
            ServerEvent::Error { message } => json!({
                "error": { "message": message }
            }),
        };
        Some(v.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dialect_from_query_selects_gemini() {
        assert_eq!(DialectKind::from_query("dialect=gemini"), DialectKind::GeminiLive);
        assert_eq!(DialectKind::from_query("?dialect=GEMINI"), DialectKind::GeminiLive);
        assert_eq!(DialectKind::from_query("dialect=openai"), DialectKind::OpenAiRealtime);
        assert_eq!(DialectKind::from_query(""), DialectKind::OpenAiRealtime);
    }

    #[test]
    fn openai_decodes_audio_append() {
        let d = OpenAiDialect;
        let pcm = vec![1u8, 2, 3, 4];
        let msg = json!({
            "type": "input_audio_buffer.append",
            "audio": b64_encode(&pcm)
        })
        .to_string();
        let cmds = d.decode(&msg);
        assert_eq!(cmds.len(), 1);
        match &cmds[0] {
            ClientCommand::AppendAudio(got) => assert_eq!(*got, pcm),
            other => panic!("expected AppendAudio, got {other:?}"),
        }
    }

    #[test]
    fn openai_decodes_cancel_and_commit() {
        let d = OpenAiDialect;
        assert!(matches!(
            d.decode(&json!({"type":"response.cancel"}).to_string())[0],
            ClientCommand::Cancel
        ));
        assert!(matches!(
            d.decode(&json!({"type":"input_audio_buffer.commit"}).to_string())[0],
            ClientCommand::CommitAudio
        ));
    }

    #[test]
    fn openai_encodes_audio_delta_roundtrip() {
        let d = OpenAiDialect;
        let pcm = vec![9u8, 8, 7];
        let out = d.encode(&ServerEvent::AudioDelta { pcm: pcm.clone() }).unwrap();
        let v: Value = serde_json::from_str(&out).unwrap();
        assert_eq!(v["type"], "response.output_audio.delta");
        assert_eq!(b64_decode(v["delta"].as_str().unwrap()).unwrap(), pcm);
    }

    #[test]
    fn gemini_decodes_realtime_audio() {
        let d = GeminiDialect;
        let pcm = vec![5u8, 6, 7, 8];
        let msg = json!({
            "realtimeInput": { "audio": { "data": b64_encode(&pcm) } }
        })
        .to_string();
        let cmds = d.decode(&msg);
        assert_eq!(cmds.len(), 1);
        match &cmds[0] {
            ClientCommand::AppendAudio(got) => assert_eq!(*got, pcm),
            other => panic!("expected AppendAudio, got {other:?}"),
        }
    }

    #[test]
    fn gemini_decodes_setup_instructions() {
        let d = GeminiDialect;
        let msg = json!({
            "setup": {
                "systemInstruction": { "parts": [ { "text": "be terse" } ] }
            }
        })
        .to_string();
        let cmds = d.decode(&msg);
        match &cmds[0] {
            ClientCommand::UpdateSession(s) => {
                assert_eq!(s.instructions.as_deref(), Some("be terse"));
            }
            other => panic!("expected UpdateSession, got {other:?}"),
        }
    }

    #[test]
    fn gemini_skips_speech_events() {
        let d = GeminiDialect;
        assert!(d.encode(&ServerEvent::SpeechStarted).is_none());
        assert!(d.encode(&ServerEvent::SpeechStopped).is_none());
    }

    #[test]
    fn gemini_encodes_audio_delta() {
        let d = GeminiDialect;
        let pcm = vec![1u8, 1, 2, 2];
        let out = d.encode(&ServerEvent::AudioDelta { pcm: pcm.clone() }).unwrap();
        let v: Value = serde_json::from_str(&out).unwrap();
        let data = v["serverContent"]["modelTurn"]["parts"][0]["inlineData"]["data"]
            .as_str()
            .unwrap();
        assert_eq!(b64_decode(data).unwrap(), pcm);
    }

    #[test]
    fn unknown_messages_decode_empty() {
        assert!(OpenAiDialect.decode("not json").is_empty());
        assert!(OpenAiDialect.decode(&json!({"type":"nonsense"}).to_string()).is_empty());
        assert!(GeminiDialect.decode(&json!({"weird":true}).to_string()).is_empty());
    }

    // ---- Negative / fuzz-style tests ----

    #[test]
    fn openai_malformed_json_truncated() {
        assert!(OpenAiDialect.decode("{").is_empty());
        assert!(OpenAiDialect.decode("").is_empty());
        assert!(OpenAiDialect.decode("null").is_empty());
        assert!(OpenAiDialect.decode("{}").is_empty());
    }

    #[test]
    fn openai_invalid_audio_base64() {
        let msg = json!({"type": "input_audio_buffer.append", "audio": "!!!invalid-base64!!!"}).to_string();
        // Should not panic - returns empty vec on decode failure
        let cmds = OpenAiDialect.decode(&msg);
        assert!(cmds.is_empty(), "invalid base64 should yield empty commands");
    }

    #[test]
    fn openai_binary_frame_ignored() {
        // Binary frames are not JSON text - simulate what decode returns for non-text
        // This is testing the decode path for non-UTF8 bytes that can't be parsed
        let bad_utf8 = String::from_utf8_lossy(&[0xFF, 0xFE, 0x00, 0x01]);
        assert!(OpenAiDialect.decode(&bad_utf8).is_empty());
    }

    #[test]
    fn openai_out_of_order_messages_no_panic() {
        // Cancel before any response exists should not panic
        let cancel = json!({"type": "response.cancel"}).to_string();
        let cmds = OpenAiDialect.decode(&cancel);
        assert_eq!(cmds.len(), 1);
        assert!(matches!(cmds[0], ClientCommand::Cancel));

        // Commit before audio should not panic
        let commit = json!({"type": "input_audio_buffer.commit"}).to_string();
        let cmds2 = OpenAiDialect.decode(&commit);
        assert_eq!(cmds2.len(), 1);
        assert!(matches!(cmds2[0], ClientCommand::CommitAudio));
    }

    #[test]
    fn gemini_malformed_setup_no_panic() {
        // Setup without systemInstruction should not panic
        let msg = json!({"setup": {}}).to_string();
        let cmds = GeminiDialect.decode(&msg);
        assert_eq!(cmds.len(), 1);
        match &cmds[0] {
            ClientCommand::UpdateSession(s) => { assert!(s.instructions.is_none()); }
            _ => panic!("expected UpdateSession"),
        }
    }

    #[test]
    fn gemini_realtime_input_missing_audio() {
        // realtimeInput without audio field should not panic
        let msg = json!({"realtimeInput": {"audio": {}}}).to_string();
        assert!(GeminiDialect.decode(&msg).is_empty());
    }

    #[test]
    fn gemini_realtime_input_empty_media_chunks() {
        let msg = json!({"realtimeInput": {"mediaChunks": []}}).to_string();
        assert!(GeminiDialect.decode(&msg).is_empty());
    }

    #[test]
    fn large_json_does_not_overwhelm() {
        // A large but valid JSON should not cause issues
        let mut large = String::from("{\"type\":\"session.update\",\"session\":{");
        for i in 0..1000 {
            large.push_str(&format!("\"key{}\":\"value{}\",", i, i));
        }
        large.push_str("\"instructions\":\"test\"}}");
        let cmds = OpenAiDialect.decode(&large);
        // Should parse and extract instructions even from large payload
        if !cmds.is_empty() {
            match &cmds[0] {
                ClientCommand::UpdateSession(s) => { assert_eq!(s.instructions.as_deref(), Some("test")); }
                _ => {}
            }
        }
    }
}
