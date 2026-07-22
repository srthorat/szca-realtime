/// Binary wire protocol for SZCA voice streaming.
///
/// Wire Format:
/// - Audio frames: Raw Int16 PCM, no framing overhead
/// - Control frames: 1-byte opcode
/// - All values in little-endian byte order

/// Control opcodes for the wire protocol.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum Opcode {
    /// Handshake / Session Init
    Handshake = 0x00,
    /// Interruption Trigger (Barge-In) — instructs gateway to flush playout buffer
    Interrupt = 0x01,
    /// End of Stream / Hangup
    EndOfStream = 0x02,
}

impl Opcode {
    /// Convert from raw byte. Returns None if invalid.
    pub fn from_byte(b: u8) -> Option<Self> {
        match b {
            0x00 => Some(Opcode::Handshake),
            0x01 => Some(Opcode::Interrupt),
            0x02 => Some(Opcode::EndOfStream),
            _ => None,
        }
    }
}

/// Audio configuration for a session.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AudioConfig {
    pub sample_rate: u32,
    pub bits_per_sample: u16,
    pub channels: u8,
}

impl Default for AudioConfig {
    fn default() -> Self {
        Self {
            sample_rate: 16000,
            bits_per_sample: 16,
            channels: 1,
        }
    }
}

impl AudioConfig {
    /// Calculate bytes per millisecond.
    pub fn bytes_per_ms(&self) -> usize {
        (self.sample_rate as usize * self.bits_per_sample as usize * self.channels as usize) / 8 / 1000
    }

    /// Calculate bytes for a given duration in milliseconds.
    pub fn bytes_for_duration_ms(&self, duration_ms: usize) -> usize {
        self.bytes_per_ms() * duration_ms
    }
}

/// A decoded frame from the wire.
#[derive(Debug, Clone)]
pub enum Frame {
    /// Audio data (raw PCM bytes)
    Audio(Vec<u8>),
    /// Control opcode
    Control(Opcode),
}

/// Decode raw bytes from WebSocket into frames.
///
/// # Arguments
/// * `data` - Raw bytes from WebSocket
/// * `config` - Audio configuration
///
/// # Returns
/// Vector of decoded frames.
pub fn decode_frames(data: &[u8], config: &AudioConfig) -> Vec<Frame> {
    if data.is_empty() {
        return Vec::new();
    }

    // A single byte is only meaningful as a control opcode. An unknown single
    // byte is too short to be a PCM sample, so it is dropped.
    if data.len() == 1 {
        return match Opcode::from_byte(data[0]) {
            Some(opcode) => vec![Frame::Control(opcode)],
            None => Vec::new(),
        };
    }

    let frame_size = config.bytes_for_duration_ms(20); // 20ms chunks

    // A leading valid opcode byte marks a control frame. Always emit the
    // control frame, then decode any trailing bytes as audio regardless of how
    // many there are (short tails included).
    if let Some(opcode) = Opcode::from_byte(data[0]) {
        let mut frames = vec![Frame::Control(opcode)];
        let remaining = &data[1..];
        for chunk in remaining.chunks(frame_size) {
            frames.push(Frame::Audio(chunk.to_vec()));
        }
        return frames;
    }

    // No opcode prefix: the entire message is raw PCM audio. The first byte is
    // treated as audio data, never folded away as an unknown opcode.
    data.chunks(frame_size)
        .map(|chunk| Frame::Audio(chunk.to_vec()))
        .collect()
}

/// Encode audio data into a binary frame for WebSocket transmission.
///
/// # Arguments
/// * `pcm_data` - Raw PCM audio bytes
///
/// # Returns
/// Ready-to-send binary frame.
pub fn encode_audio_frame(pcm_data: &[u8]) -> Vec<u8> {
    pcm_data.to_vec()
}

/// Encode a control opcode into a binary frame.
///
/// # Arguments
/// * `opcode` - Control opcode to encode
///
/// # Returns
/// Single-byte frame.
pub fn encode_control_frame(opcode: Opcode) -> Vec<u8> {
    vec![opcode as u8]
}

/// Validate that audio data has correct alignment for the given config.
pub fn validate_audio_alignment(data: &[u8], config: &AudioConfig) -> bool {
    let bytes_per_sample = config.bits_per_sample as usize / 8;
    let frame_size = config.bytes_for_duration_ms(20); // 20ms chunks
    data.len() % bytes_per_sample == 0 && (data.len() % frame_size == 0 || data.len() < frame_size)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_opcode_from_byte_valid() {
        assert_eq!(Opcode::from_byte(0x00), Some(Opcode::Handshake));
        assert_eq!(Opcode::from_byte(0x01), Some(Opcode::Interrupt));
        assert_eq!(Opcode::from_byte(0x02), Some(Opcode::EndOfStream));
    }

    #[test]
    fn test_opcode_from_byte_invalid() {
        assert_eq!(Opcode::from_byte(0x03), None);
        assert_eq!(Opcode::from_byte(0xFF), None);
        assert_eq!(Opcode::from_byte(0x10), None);
    }

    #[test]
    fn test_opcode_values() {
        assert_eq!(Opcode::Handshake as u8, 0x00);
        assert_eq!(Opcode::Interrupt as u8, 0x01);
        assert_eq!(Opcode::EndOfStream as u8, 0x02);
    }

    #[test]
    fn test_audio_config_default() {
        let config = AudioConfig::default();
        assert_eq!(config.sample_rate, 16000);
        assert_eq!(config.bits_per_sample, 16);
        assert_eq!(config.channels, 1);
    }

    #[test]
    fn test_audio_config_bytes_per_ms() {
        let config = AudioConfig::default();
        // 16000 Hz * 16 bits * 1 channel / 8 bits/byte / 1000 ms = 32 bytes/ms
        assert_eq!(config.bytes_per_ms(), 32);
    }

    #[test]
    fn test_audio_config_bytes_for_duration_ms() {
        let config = AudioConfig::default();
        // 20ms * 32 bytes/ms = 640 bytes
        assert_eq!(config.bytes_for_duration_ms(20), 640);
    }

    #[test]
    fn test_audio_config_stereo() {
        let config = AudioConfig {
            sample_rate: 44100,
            bits_per_sample: 16,
            channels: 2,
        };
        // 44100 * 16 * 2 / 8 / 1000 = 176.4 bytes/ms
        assert_eq!(config.bytes_per_ms(), 176);
    }

    #[test]
    fn test_decode_frames_empty() {
        let config = AudioConfig::default();
        let frames = decode_frames(&[], &config);
        assert!(frames.is_empty());
    }

    #[test]
    fn test_decode_frames_handshake() {
        let config = AudioConfig::default();
        let frames = decode_frames(&[0x00], &config);
        assert_eq!(frames.len(), 1);
        assert!(matches!(frames[0], Frame::Control(Opcode::Handshake)));
    }

    #[test]
    fn test_decode_frames_interrupt() {
        let config = AudioConfig::default();
        let frames = decode_frames(&[0x01], &config);
        assert_eq!(frames.len(), 1);
        assert!(matches!(frames[0], Frame::Control(Opcode::Interrupt)));
    }

    #[test]
    fn test_decode_frames_end_of_stream() {
        let config = AudioConfig::default();
        let frames = decode_frames(&[0x02], &config);
        assert_eq!(frames.len(), 1);
        assert!(matches!(frames[0], Frame::Control(Opcode::EndOfStream)));
    }

    #[test]
    fn test_decode_frames_control_plus_short_tail() {
        // Regression (H5): a control opcode followed by a short (< 20ms) data
        // tail must still emit the control frame AND the audio tail, not drop
        // the opcode.
        let config = AudioConfig::default();
        let mut data = vec![0x01u8]; // Interrupt opcode
        data.extend_from_slice(&[0xABu8; 10]); // 10-byte short tail

        let frames = decode_frames(&data, &config);
        assert_eq!(frames.len(), 2);
        assert!(matches!(frames[0], Frame::Control(Opcode::Interrupt)));
        match &frames[1] {
            Frame::Audio(pcm) => assert_eq!(pcm.len(), 10),
            _ => panic!("Expected audio tail after control opcode"),
        }
    }

    #[test]
    fn test_decode_frames_unknown_first_byte_is_audio() {
        // An unknown leading byte must be treated as PCM audio, not folded away.
        let config = AudioConfig::default();
        let mut data = vec![0xFFu8]; // Not a valid opcode
        data.extend_from_slice(&[0x00u8; 639]); // total 640 bytes
        let frames = decode_frames(&data, &config);
        assert_eq!(frames.len(), 1);
        match &frames[0] {
            Frame::Audio(pcm) => {
                assert_eq!(pcm.len(), 640);
                assert_eq!(pcm[0], 0xFF); // first byte preserved as audio
            }
            _ => panic!("Expected audio frame"),
        }
    }

    #[test]
    fn test_decode_frames_invalid_opcode() {
        let config = AudioConfig::default();
        let frames = decode_frames(&[0x03], &config);
        assert!(frames.is_empty());
    }

    #[test]
    fn test_decode_frames_audio_only() {
        let config = AudioConfig::default();
        // 640 bytes = 20ms of 16kHz 16-bit mono audio
        let audio = vec![0xABu8; 640];
        let frames = decode_frames(&audio, &config);
        assert_eq!(frames.len(), 1);
        assert!(matches!(frames[0], Frame::Audio(_)));
    }

    #[test]
    fn test_decode_frames_audio_multiple_chunks() {
        let config = AudioConfig::default();
        // 1280 bytes = 40ms of audio (2 chunks)
        let audio = vec![0xABu8; 1280];
        let frames = decode_frames(&audio, &config);
        assert_eq!(frames.len(), 2);
    }

    #[test]
    fn test_decode_frames_partial_chunk() {
        let config = AudioConfig::default();
        // 320 bytes = 10ms (less than 20ms frame)
        let audio = vec![0xABu8; 320];
        let frames = decode_frames(&audio, &config);
        assert_eq!(frames.len(), 1);
        assert!(matches!(frames[0], Frame::Audio(_)));
    }

    #[test]
    fn test_encode_audio_frame() {
        let data = vec![0x01, 0x02, 0x03];
        let encoded = encode_audio_frame(&data);
        assert_eq!(encoded, data);
    }

    #[test]
    fn test_encode_control_frame() {
        let encoded = encode_control_frame(Opcode::Handshake);
        assert_eq!(encoded, vec![0x00]);

        let encoded = encode_control_frame(Opcode::Interrupt);
        assert_eq!(encoded, vec![0x01]);

        let encoded = encode_control_frame(Opcode::EndOfStream);
        assert_eq!(encoded, vec![0x02]);
    }

    #[test]
    fn test_validate_audio_alignment_valid() {
        let config = AudioConfig::default();
        let audio = vec![0u8; 640]; // Exactly 20ms
        assert!(validate_audio_alignment(&audio, &config));
    }

    #[test]
    fn test_validate_audio_alignment_partial() {
        let config = AudioConfig::default();
        let audio = vec![0u8; 320]; // 10ms, still aligned to sample boundary
        assert!(validate_audio_alignment(&audio, &config));
    }

    #[test]
    fn test_validate_audio_alignment_misaligned() {
        let config = AudioConfig::default();
        let audio = vec![0u8; 641]; // Not aligned to 2 bytes (sample)
        assert!(!validate_audio_alignment(&audio, &config));
    }

    #[test]
    fn test_roundtrip_encode_decode_audio() {
        let config = AudioConfig::default();
        let original = vec![0xABu8; 640];
        let encoded = encode_audio_frame(&original);
        let frames = decode_frames(&encoded, &config);

        assert_eq!(frames.len(), 1);
        if let Frame::Audio(decoded) = &frames[0] {
            assert_eq!(*decoded, original);
        } else {
            panic!("Expected audio frame");
        }
    }

    #[test]
    fn test_roundtrip_encode_decode_control() {
        let config = AudioConfig::default();
        let encoded = encode_control_frame(Opcode::Interrupt);
        let frames = decode_frames(&encoded, &config);

        assert_eq!(frames.len(), 1);
        assert!(matches!(frames[0], Frame::Control(Opcode::Interrupt)));
    }

    #[test]
    fn test_decode_frames_large_audio() {
        let config = AudioConfig::default();
        // 1 second of audio = 32000 bytes
        let audio = vec![0xABu8; 32000];
        let frames = decode_frames(&audio, &config);
        assert_eq!(frames.len(), 50); // 1000ms / 20ms = 50 frames
    }
}
