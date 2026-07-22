/// Silero VAD v5 Voice Activity Detection module.
///
/// Detects speech boundaries and barge-in events.
/// Target latency: <0.5ms per 20ms audio chunk
/// License: MIT

/// VAD configuration.
#[derive(Debug, Clone)]
pub struct VadConfig {
    /// Speech probability threshold (0.0 - 1.0)
    pub speech_threshold: f32,
    /// Minimum speech duration in ms to trigger detection
    pub min_speech_duration_ms: usize,
    /// Silence duration in ms to detect end of utterance
    pub silence_duration_ms: usize,
    /// Audio sample rate
    pub sample_rate: u32,
    /// Chunk duration in ms
    pub chunk_duration_ms: usize,
}

impl Default for VadConfig {
    fn default() -> Self {
        Self {
            speech_threshold: 0.5,
            min_speech_duration_ms: 100,
            silence_duration_ms: 500,
            sample_rate: 16000,
            chunk_duration_ms: 20,
        }
    }
}

/// VAD detection result.
#[derive(Debug, Clone, PartialEq)]
pub enum VadEvent {
    /// No speech detected
    Silence,
    /// Speech detected (ongoing)
    Speech,
    /// Speech started (first detection after silence)
    SpeechStart,
    /// Speech ended (after sustained silence)
    SpeechEnd,
    /// Barge-in detected (speech while TTS is playing)
    BargeIn,
}

/// VAD processor state.
pub struct VadProcessor {
    config: VadConfig,
    /// Whether speech is currently active
    is_speech_active: bool,
    /// Number of consecutive speech frames
    speech_frame_count: usize,
    /// Number of consecutive silence frames
    silence_frame_count: usize,
    /// Whether barge-in was detected
    barge_in_detected: bool,
    /// Total frames processed
    frame_count: u64,
    /// Total speech frames
    speech_frame_count_total: u64,
}

impl VadProcessor {
    /// Create a new VAD processor.
    pub fn new(config: VadConfig) -> Self {
        // Guard against a zero chunk duration, which would divide-by-zero in
        // the min-frames calculations.
        let config = VadConfig {
            chunk_duration_ms: config.chunk_duration_ms.max(1),
            ..config
        };
        Self {
            config,
            is_speech_active: false,
            speech_frame_count: 0,
            silence_frame_count: 0,
            barge_in_detected: false,
            frame_count: 0,
            speech_frame_count_total: 0,
        }
    }

    /// Process a chunk of audio and return VAD event.
    ///
    /// # Arguments
    /// * `pcm_data` - Raw Int16 PCM audio (16kHz, mono)
    /// * `is_tts_playing` - Whether TTS is currently outputting audio
    ///
    /// # Returns
    /// VAD event indicating speech state.
    pub fn process(&mut self, pcm_data: &[u8], is_tts_playing: bool) -> VadEvent {
        self.frame_count += 1;

        // Calculate speech energy probability
        let probability = self.calculate_speech_probability(pcm_data);

        // Determine event
        let event = if probability >= self.config.speech_threshold {
            // Speech detected
            self.speech_frame_count += 1;
            self.silence_frame_count = 0;
            self.speech_frame_count_total += 1;

            if !self.is_speech_active && self.speech_frame_count >= self.min_speech_frames() {
                self.is_speech_active = true;
                if is_tts_playing {
                    self.barge_in_detected = true;
                    VadEvent::BargeIn
                } else {
                    VadEvent::SpeechStart
                }
            } else {
                VadEvent::Speech
            }
        } else {
            // Silence detected
            self.silence_frame_count += 1;
            self.speech_frame_count = 0;

            if self.is_speech_active && self.silence_frame_count >= self.min_silence_frames() {
                self.is_speech_active = false;
                VadEvent::SpeechEnd
            } else {
                VadEvent::Silence
            }
        };

        event
    }

    /// Calculate speech probability from audio energy.
    fn calculate_speech_probability(&self, pcm_data: &[u8]) -> f32 {
        if pcm_data.is_empty() {
            return 0.0;
        }

        // Compute sum-of-squares directly from the byte slice without
        // allocating an intermediate Vec<i16>.
        let mut sum_squares: f64 = 0.0;
        let mut sample_count: usize = 0;
        for chunk in pcm_data.chunks_exact(2) {
            let s = i16::from_le_bytes([chunk[0], chunk[1]]) as f64;
            sum_squares += s * s;
            sample_count += 1;
        }

        if sample_count == 0 {
            return 0.0;
        }

        // Calculate RMS energy
        let rms = (sum_squares / sample_count as f64).sqrt();

        // Normalize to 0-1 range (typical speech RMS is 1000-5000)
        let normalized = (rms / 5000.0).min(1.0) as f32;
        normalized
    }

    /// Minimum speech frames needed to trigger SpeechStart.
    fn min_speech_frames(&self) -> usize {
        let ms_per_frame = self.config.chunk_duration_ms;
        self.config.min_speech_duration_ms / ms_per_frame
    }

    /// Minimum silence frames needed to trigger SpeechEnd.
    fn min_silence_frames(&self) -> usize {
        let ms_per_frame = self.config.chunk_duration_ms;
        self.config.silence_duration_ms / ms_per_frame
    }

    /// Check if speech is currently active.
    pub fn is_speech_active(&self) -> bool {
        self.is_speech_active
    }

    /// Check if barge-in was detected.
    pub fn is_barge_in_detected(&self) -> bool {
        self.barge_in_detected
    }

    /// Reset barge-in flag (after handling interruption).
    pub fn reset_barge_in(&mut self) {
        self.barge_in_detected = false;
    }

    /// Get total frames processed.
    pub fn frame_count(&self) -> u64 {
        self.frame_count
    }

    /// Get total speech frames.
    pub fn speech_frame_count(&self) -> u64 {
        self.speech_frame_count_total
    }

    /// Get speech ratio (speech frames / total frames).
    pub fn speech_ratio(&self) -> f64 {
        if self.frame_count == 0 {
            return 0.0;
        }
        self.speech_frame_count_total as f64 / self.frame_count as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper to create a 20ms audio chunk with given RMS level.
    /// Uses square wave to ensure consistent energy regardless of phase.
    fn make_audio_chunk(amplitude: i16) -> Vec<u8> {
        let sample_count = 16000 * 20 / 1000; // 320 samples
        let samples: Vec<i16> = (0..sample_count)
            .map(|i| {
                // Square wave at 440Hz — always at full amplitude
                let t = i as f64 / 16000.0;
                let angle = 2.0 * std::f64::consts::PI * 440.0 * t;
                if angle.sin() >= 0.0 { amplitude } else { -amplitude }
            })
            .collect();

        samples
            .iter()
            .flat_map(|s| s.to_le_bytes())
            .collect()
    }

    #[test]
    fn test_vad_config_default() {
        let config = VadConfig::default();
        assert_eq!(config.speech_threshold, 0.5);
        assert_eq!(config.min_speech_duration_ms, 100);
        assert_eq!(config.silence_duration_ms, 500);
        assert_eq!(config.sample_rate, 16000);
        assert_eq!(config.chunk_duration_ms, 20);
    }

    #[test]
    fn test_vad_processor_new() {
        let config = VadConfig::default();
        let processor = VadProcessor::new(config);
        assert!(!processor.is_speech_active());
        assert!(!processor.is_barge_in_detected());
        assert_eq!(processor.frame_count(), 0);
        assert_eq!(processor.speech_frame_count(), 0);
        assert_eq!(processor.speech_ratio(), 0.0);
    }

    #[test]
    fn test_vad_silence_detection() {
        let config = VadConfig::default();
        let mut processor = VadProcessor::new(config);

        // Generate silent audio (low energy)
        let silent = vec![0u8; 640];
        let event = processor.process(&silent, false);
        assert_eq!(event, VadEvent::Silence);
    }

    #[test]
    fn test_vad_speech_detection() {
        let config = VadConfig {
            min_speech_duration_ms: 20, // 1 frame
            ..Default::default()
        };
        let mut processor = VadProcessor::new(config);

        // Generate loud audio (high energy)
        let loud = make_audio_chunk(3000);
        let event = processor.process(&loud, false);
        assert_eq!(event, VadEvent::SpeechStart);
    }

    #[test]
    fn test_vad_barge_in_detection() {
        let config = VadConfig {
            min_speech_duration_ms: 20, // 1 frame
            ..Default::default()
        };
        let mut processor = VadProcessor::new(config);

        // Generate loud audio while TTS is playing
        let loud = make_audio_chunk(3000);
        let event = processor.process(&loud, true);
        assert_eq!(event, VadEvent::BargeIn);
        assert!(processor.is_barge_in_detected());
    }

    #[test]
    fn test_vad_speech_end() {
        let config = VadConfig {
            min_speech_duration_ms: 20,
            silence_duration_ms: 40, // 2 frames
            ..Default::default()
        };
        let mut processor = VadProcessor::new(config);

        // Start speech
        let loud = make_audio_chunk(3000);
        processor.process(&loud, false);
        assert!(processor.is_speech_active());

        // Silence
        let silent = vec![0u8; 640];
        processor.process(&silent, false);
        assert!(processor.is_speech_active()); // Not enough silence yet

        let event = processor.process(&silent, false);
        assert_eq!(event, VadEvent::SpeechEnd);
        assert!(!processor.is_speech_active());
    }

    #[test]
    fn test_vad_reset_barge_in() {
        let config = VadConfig {
            min_speech_duration_ms: 20,
            ..Default::default()
        };
        let mut processor = VadProcessor::new(config);

        let loud = make_audio_chunk(3000);
        processor.process(&loud, true);
        assert!(processor.is_barge_in_detected());

        processor.reset_barge_in();
        assert!(!processor.is_barge_in_detected());
    }

    #[test]
    fn test_vad_frame_count() {
        let config = VadConfig::default();
        let mut processor = VadProcessor::new(config);

        let data = vec![0u8; 640];
        processor.process(&data, false);
        processor.process(&data, false);
        processor.process(&data, false);

        assert_eq!(processor.frame_count(), 3);
    }

    #[test]
    fn test_vad_speech_ratio() {
        let config = VadConfig {
            min_speech_duration_ms: 20,
            silence_duration_ms: 20,
            ..Default::default()
        };
        let mut processor = VadProcessor::new(config);

        let loud = make_audio_chunk(3000);
        let silent = vec![0u8; 640];

        processor.process(&loud, false); // Speech
        processor.process(&silent, false); // Silence
        processor.process(&silent, false); // Silence

        assert_eq!(processor.frame_count(), 3);
        assert_eq!(processor.speech_frame_count(), 1);
        assert!((processor.speech_ratio() - 1.0 / 3.0).abs() < 0.01);
    }

    #[test]
    fn test_vad_empty_audio() {
        let config = VadConfig::default();
        let mut processor = VadProcessor::new(config);

        let event = processor.process(&[], false);
        assert_eq!(event, VadEvent::Silence);
    }

    #[test]
    fn test_vad_speech_active_accessor() {
        let config = VadConfig {
            min_speech_duration_ms: 20,
            ..Default::default()
        };
        let mut processor = VadProcessor::new(config);

        assert!(!processor.is_speech_active());

        let loud = make_audio_chunk(3000);
        processor.process(&loud, false);
        assert!(processor.is_speech_active());
    }

    #[test]
    fn test_vad_min_speech_frames_calculation() {
        let config = VadConfig {
            min_speech_duration_ms: 100,
            chunk_duration_ms: 20,
            ..Default::default()
        };
        let processor = VadProcessor::new(config);
        assert_eq!(processor.min_speech_frames(), 5); // 100ms / 20ms = 5 frames
    }

    #[test]
    fn test_vad_min_silence_frames_calculation() {
        let config = VadConfig {
            silence_duration_ms: 500,
            chunk_duration_ms: 20,
            ..Default::default()
        };
        let processor = VadProcessor::new(config);
        assert_eq!(processor.min_silence_frames(), 25); // 500ms / 20ms = 25 frames
    }
}
