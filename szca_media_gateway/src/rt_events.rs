/// Dialect-agnostic realtime event model.
///
/// The realtime WebSocket API speaks more than one wire "dialect" (OpenAI
/// Realtime, Gemini Live). To avoid coupling the session/pipeline logic to any
/// one vendor's JSON shape, everything internal is expressed in terms of these
/// neutral types. A [`crate::rt_protocol::Dialect`] translates between a wire
/// dialect and these types in both directions:
///
///   inbound  : wire JSON  --Dialect::decode-->  Vec<ClientCommand>
///   outbound : ServerEvent --Dialect::encode-->  wire JSON
///
/// Audio is always carried as raw little-endian 16-bit PCM, mono, 16 kHz —
/// the format the VAD and (Phase 2) STT expect. Dialect adapters are
/// responsible for base64 decode/encode at the wire boundary.

/// Per-session settings a client may configure at/after connect.
#[derive(Debug, Clone, Default)]
pub struct SessionSettings {
    /// System/developer instructions for the LLM stage.
    pub instructions: Option<String>,
    /// TTS voice identifier (e.g. `af_heart`).
    pub voice: Option<String>,
    /// When true (default), server-side VAD decides turn boundaries and
    /// auto-triggers a response on end-of-speech. When false, the client must
    /// explicitly commit the input buffer / request a response.
    pub auto_turn_detection: Option<bool>,
}

/// A normalized command decoded from an inbound client message.
#[derive(Debug, Clone)]
pub enum ClientCommand {
    /// Update session-level settings.
    UpdateSession(SessionSettings),
    /// Append captured microphone audio (PCM16 mono 16 kHz).
    AppendAudio(Vec<u8>),
    /// Manually mark end-of-utterance (client-side turn detection).
    CommitAudio,
    /// Explicitly ask the server to produce a response now.
    CreateResponse,
    /// Cancel the in-flight response (barge-in / interrupt).
    Cancel,
    /// Client is done; close the session.
    Hangup,
}

/// A normalized event to be delivered to the client.
#[derive(Debug, Clone)]
pub enum ServerEvent {
    /// Session accepted and ready.
    SessionCreated { session_id: String },
    /// Server VAD detected the user started speaking.
    SpeechStarted,
    /// Server VAD detected the user stopped speaking (end of turn).
    SpeechStopped,
    /// Incremental transcription of the user's speech.
    TranscriptDelta { text: String },
    /// Final transcription of the user's utterance.
    TranscriptDone { text: String },
    /// Incremental assistant text (LLM token(s)).
    TextDelta { text: String },
    /// Final assistant text for this response.
    TextDone { text: String },
    /// A chunk of synthesized assistant audio (PCM16 mono).
    AudioDelta { pcm: Vec<u8> },
    /// No more audio for this response.
    AudioDone,
    /// The current response was cancelled (barge-in).
    ResponseCancelled,
    /// The response turn is fully complete.
    ResponseDone,
    /// A recoverable error to surface to the client.
    Error { message: String },
}
