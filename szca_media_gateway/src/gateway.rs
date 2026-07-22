/// WebSocket gateway server for SZCA voice streaming.
///
/// Handles:
/// - WebSocket upgrade and connection management
/// - Binary frame decoding (audio + control)
/// - Session lifecycle (create, active, barge-in, end)
/// - Egress task (engine → client audio streaming)

use crate::protocol::{self, AudioConfig, Frame, Opcode};
use crate::session::{Session, SessionState};

/// Gateway configuration.
#[derive(Debug, Clone)]
pub struct GatewayConfig {
    /// Listen address
    pub listen_addr: String,
    /// Listen port
    pub port: u16,
    /// Maximum concurrent sessions
    pub max_sessions: usize,
    /// Audio configuration
    pub audio_config: AudioConfig,
}

impl Default for GatewayConfig {
    fn default() -> Self {
        Self {
            listen_addr: "0.0.0.0".to_string(),
            port: 3000,
            max_sessions: 1000,
            audio_config: AudioConfig::default(),
        }
    }
}

/// Gateway error types.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GatewayError {
    /// Session limit reached
    SessionLimitReached,
    /// Invalid session state
    InvalidSessionState(String),
    /// Protocol error
    ProtocolError(String),
}

impl std::fmt::Display for GatewayError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GatewayError::SessionLimitReached => write!(f, "Session limit reached"),
            GatewayError::InvalidSessionState(msg) => write!(f, "Invalid session state: {}", msg),
            GatewayError::ProtocolError(msg) => write!(f, "Protocol error: {}", msg),
        }
    }
}

impl std::error::Error for GatewayError {}

/// Process a single incoming WebSocket message.
///
/// Returns a list of frames to forward to the inference engine.
pub fn process_incoming_message(
    data: &[u8],
    session: &mut Session,
    audio_config: &AudioConfig,
) -> Result<Vec<Frame>, GatewayError> {
    // Decode frames from raw bytes
    let frames = protocol::decode_frames(data, audio_config);

    let mut engine_frames = Vec::new();

    for frame in frames {
        match frame {
            Frame::Control(opcode) => {
                match opcode {
                    Opcode::Handshake => {
                        if session.state() == SessionState::Created {
                            session.activate().map_err(|e| {
                                GatewayError::InvalidSessionState(e.to_string())
                            })?;
                        }
                    }
                    Opcode::Interrupt => {
                        session.barge_in();
                        session.record_bytes_in(data.len() as u64);
                    }
                    Opcode::EndOfStream => {
                        session.end();
                    }
                }
            }
            Frame::Audio(pcm_data) => {
                if session.state() == SessionState::Active {
                    session.record_bytes_in(pcm_data.len() as u64);
                    engine_frames.push(Frame::Audio(pcm_data));
                }
            }
        }
    }

    Ok(engine_frames)
}

/// Validate WebSocket upgrade request.
pub fn validate_upgrade(path: &str) -> bool {
    path == "/v1/realtime"
}

/// Format an audio chunk as a WebSocket binary frame.
pub fn format_audio_frame(pcm_data: &[u8]) -> Vec<u8> {
    protocol::encode_audio_frame(pcm_data)
}

/// Format a control opcode as a WebSocket binary frame.
pub fn format_control_frame(opcode: Opcode) -> Vec<u8> {
    protocol::encode_control_frame(opcode)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::session::SessionConfig;

    #[test]
    fn test_gateway_config_default() {
        let config = GatewayConfig::default();
        assert_eq!(config.listen_addr, "0.0.0.0");
        assert_eq!(config.port, 3000);
        assert_eq!(config.max_sessions, 1000);
    }

    #[test]
    fn test_process_incoming_handshake() {
        let config = SessionConfig::default();
        let mut session = Session::new(config);
        let audio_config = AudioConfig::default();

        let data = vec![0x00]; // Handshake opcode
        let frames = process_incoming_message(&data, &mut session, &audio_config).unwrap();

        assert!(frames.is_empty());
        assert_eq!(session.state(), SessionState::Active);
    }

    #[test]
    fn test_process_incoming_audio() {
        let config = SessionConfig::default();
        let mut session = Session::new(config);
        session.activate().unwrap();
        let audio_config = AudioConfig::default();

        let data = vec![0xABu8; 640]; // 20ms audio
        let frames = process_incoming_message(&data, &mut session, &audio_config).unwrap();

        assert_eq!(frames.len(), 1);
        assert!(matches!(frames[0], Frame::Audio(_)));
        assert_eq!(session.stats().bytes_in, 640);
    }

    #[test]
    fn test_process_incoming_interrupt() {
        let config = SessionConfig::default();
        let mut session = Session::new(config);
        session.activate().unwrap();
        let audio_config = AudioConfig::default();

        let data = vec![0x01]; // Interrupt opcode
        let frames = process_incoming_message(&data, &mut session, &audio_config).unwrap();

        assert!(frames.is_empty());
        assert!(session.is_cancelled());
    }

    #[test]
    fn test_process_incoming_end_of_stream() {
        let config = SessionConfig::default();
        let mut session = Session::new(config);
        session.activate().unwrap();
        let audio_config = AudioConfig::default();

        let data = vec![0x02]; // End of stream
        let frames = process_incoming_message(&data, &mut session, &audio_config).unwrap();

        assert!(frames.is_empty());
        assert_eq!(session.state(), SessionState::Ended);
    }

    #[test]
    fn test_process_incoming_empty_data() {
        let config = SessionConfig::default();
        let mut session = Session::new(config);
        let audio_config = AudioConfig::default();

        let data = vec![];
        let frames = process_incoming_message(&data, &mut session, &audio_config).unwrap();
        assert!(frames.is_empty());
    }

    #[test]
    fn test_process_incoming_audio_before_handshake() {
        let config = SessionConfig::default();
        let mut session = Session::new(config);
        let audio_config = AudioConfig::default();

        let data = vec![0xABu8; 640]; // Audio before handshake
        let frames = process_incoming_message(&data, &mut session, &audio_config).unwrap();

        // Audio should be dropped (session not active)
        assert!(frames.is_empty());
    }

    #[test]
    fn test_validate_upgrade_valid() {
        assert!(validate_upgrade("/v1/realtime"));
    }

    #[test]
    fn test_validate_upgrade_invalid() {
        assert!(!validate_upgrade("/v1/chat"));
        assert!(!validate_upgrade("/"));
        assert!(!validate_upgrade("/v1/realtime/stream"));
    }

    #[test]
    fn test_format_audio_frame() {
        let pcm = vec![0x01, 0x02, 0x03, 0x04];
        let frame = format_audio_frame(&pcm);
        assert_eq!(frame, pcm);
    }

    #[test]
    fn test_format_control_frame() {
        let frame = format_control_frame(Opcode::Interrupt);
        assert_eq!(frame, vec![0x01]);
    }

    #[test]
    fn test_gateway_error_display() {
        let err = GatewayError::SessionLimitReached;
        assert!(format!("{}", err).contains("Session limit reached"));

        let err = GatewayError::InvalidSessionState("test".to_string());
        assert!(format!("{}", err).contains("test"));

        let err = GatewayError::ProtocolError("bad frame".to_string());
        assert!(format!("{}", err).contains("bad frame"));
    }

    #[test]
    fn test_process_incoming_multiple_audio_chunks() {
        let config = SessionConfig::default();
        let mut session = Session::new(config);
        session.activate().unwrap();
        let audio_config = AudioConfig::default();

        // 1280 bytes = 2 chunks of 640
        let data = vec![0xABu8; 1280];
        let frames = process_incoming_message(&data, &mut session, &audio_config).unwrap();

        assert_eq!(frames.len(), 2);
        assert_eq!(session.stats().bytes_in, 1280);
    }

    #[test]
    fn test_process_handshake_then_audio() {
        let config = SessionConfig::default();
        let mut session = Session::new(config);
        let audio_config = AudioConfig::default();

        // Handshake first
        let data = vec![0x00];
        process_incoming_message(&data, &mut session, &audio_config).unwrap();
        assert_eq!(session.state(), SessionState::Active);

        // Then audio
        let data = vec![0xABu8; 640];
        let frames = process_incoming_message(&data, &mut session, &audio_config).unwrap();
        assert_eq!(frames.len(), 1);
    }
}
