/// Session management for voice streaming connections.
///
/// Each WebSocket connection maps to a Session that owns:
/// - Audio state (config, counters)
/// - VAD state (speech detection, barge-in)
/// - IPC channel (shared memory link to inference engine)
/// - Cancellation flag (atomic interrupt)

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use uuid::Uuid;

/// Session configuration.
#[derive(Debug, Clone)]
pub struct SessionConfig {
    /// IPC channel prefix for this session
    pub ipc_prefix: String,
    /// Audio sample rate
    pub sample_rate: u32,
    /// Audio chunk duration in ms
    pub chunk_duration_ms: usize,
}

impl Default for SessionConfig {
    fn default() -> Self {
        Self {
            ipc_prefix: format!("/dev/shm/szca_{}", Uuid::new_v4()),
            sample_rate: 16000,
            chunk_duration_ms: 20,
        }
    }
}

/// Session state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SessionState {
    /// Session created, waiting for handshake
    Created,
    /// Handshake received, active streaming
    Active,
    /// Session paused (e.g., barge-in in progress)
    Paused,
    /// Session ended
    Ended,
}

/// Audio statistics for a session.
#[derive(Debug, Clone, Default)]
pub struct AudioStats {
    /// Total audio bytes received
    pub bytes_in: u64,
    /// Total audio bytes sent
    pub bytes_out: u64,
    /// Number of STT partial results
    pub stt_partials: u32,
    /// Number of STT final results
    pub stt_finals: u32,
    /// Number of LLM tokens generated
    pub llm_tokens: u32,
    /// Number of TTS audio chunks
    pub tts_chunks: u32,
    /// Last latency measurement in ms
    pub last_latency_ms: f64,
}

/// A voice streaming session.
pub struct Session {
    /// Unique session identifier
    id: String,
    /// Current state
    state: SessionState,
    /// Session configuration
    config: SessionConfig,
    /// Audio statistics
    stats: AudioStats,
    /// Atomic cancellation flag (for barge-in)
    cancel_flag: Arc<AtomicBool>,
    /// Whether TTS is currently playing
    tts_playing: bool,
}

impl Session {
    /// Create a new session.
    pub fn new(config: SessionConfig) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            state: SessionState::Created,
            config,
            stats: AudioStats::default(),
            cancel_flag: Arc::new(AtomicBool::new(false)),
            tts_playing: false,
        }
    }

    /// Get the session ID.
    pub fn id(&self) -> &str {
        &self.id
    }

    /// Get the current state.
    pub fn state(&self) -> SessionState {
        self.state
    }

    /// Transition to Active state (after handshake).
    pub fn activate(&mut self) -> Result<(), SessionError> {
        if self.state != SessionState::Created {
            return Err(SessionError::InvalidStateTransition {
                from: self.state,
                to: SessionState::Active,
            });
        }
        self.state = SessionState::Active;
        Ok(())
    }

    /// Transition to Paused state.
    pub fn pause(&mut self) -> Result<(), SessionError> {
        if self.state != SessionState::Active {
            return Err(SessionError::InvalidStateTransition {
                from: self.state,
                to: SessionState::Paused,
            });
        }
        self.state = SessionState::Paused;
        Ok(())
    }

    /// Resume from Paused to Active state.
    pub fn resume(&mut self) -> Result<(), SessionError> {
        if self.state != SessionState::Paused {
            return Err(SessionError::InvalidStateTransition {
                from: self.state,
                to: SessionState::Active,
            });
        }
        self.state = SessionState::Active;
        Ok(())
    }

    /// End the session.
    pub fn end(&mut self) {
        self.state = SessionState::Ended;
        self.cancel_flag.store(true, Ordering::Relaxed);
    }

    /// Trigger barge-in (cancel current TTS).
    pub fn barge_in(&mut self) {
        self.cancel_flag.store(true, Ordering::Relaxed);
        self.tts_playing = false;
    }

    /// Reset barge-in flag.
    pub fn reset_barge_in(&mut self) {
        self.cancel_flag.store(false, Ordering::Relaxed);
    }

    /// Check if cancellation is requested.
    pub fn is_cancelled(&self) -> bool {
        self.cancel_flag.load(Ordering::Relaxed)
    }

    /// Get the cancel flag Arc (for sharing with async tasks).
    pub fn cancel_flag(&self) -> Arc<AtomicBool> {
        Arc::clone(&self.cancel_flag)
    }

    /// Update audio statistics.
    pub fn record_bytes_in(&mut self, bytes: u64) {
        self.stats.bytes_in = self.stats.bytes_in.saturating_add(bytes);
    }

    pub fn record_bytes_out(&mut self, bytes: u64) {
        self.stats.bytes_out = self.stats.bytes_out.saturating_add(bytes);
    }

    pub fn record_stt_partial(&mut self) {
        self.stats.stt_partials = self.stats.stt_partials.saturating_add(1);
    }

    pub fn record_stt_final(&mut self) {
        self.stats.stt_finals = self.stats.stt_finals.saturating_add(1);
    }

    pub fn record_llm_token(&mut self) {
        self.stats.llm_tokens = self.stats.llm_tokens.saturating_add(1);
    }

    pub fn record_tts_chunk(&mut self) {
        self.stats.tts_chunks = self.stats.tts_chunks.saturating_add(1);
    }

    pub fn record_latency(&mut self, latency_ms: f64) {
        self.stats.last_latency_ms = latency_ms;
    }

    /// Get audio statistics.
    pub fn stats(&self) -> &AudioStats {
        &self.stats
    }

    /// Set TTS playing state.
    pub fn set_tts_playing(&mut self, playing: bool) {
        self.tts_playing = playing;
    }

    /// Check if TTS is playing.
    pub fn is_tts_playing(&self) -> bool {
        self.tts_playing
    }

    /// Get session configuration.
    pub fn config(&self) -> &SessionConfig {
        &self.config
    }
}

/// Session errors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SessionError {
    /// Invalid state transition
    InvalidStateTransition {
        from: SessionState,
        to: SessionState,
    },
}

impl std::fmt::Display for SessionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SessionError::InvalidStateTransition { from, to } => {
                write!(f, "Invalid state transition: {:?} → {:?}", from, to)
            }
        }
    }
}

impl std::error::Error for SessionError {}

/// Session manager for handling multiple concurrent sessions.
///
/// Thread-safe via an atomic active-session counter, so it can be shared
/// behind an `Arc` across connection handlers for admission control.
pub struct SessionManager {
    /// Maximum concurrent sessions
    max_sessions: usize,
    /// Active session count (atomic for interior mutability)
    active_count: AtomicUsize,
}

impl SessionManager {
    /// Create a new session manager.
    pub fn new(max_sessions: usize) -> Self {
        Self {
            max_sessions,
            active_count: AtomicUsize::new(0),
        }
    }

    /// Check if a new session can be created.
    pub fn can_create(&self) -> bool {
        self.active_count.load(Ordering::Relaxed) < self.max_sessions
    }

    /// Register a new session.
    ///
    /// Atomically reserves a slot; returns an error if the limit is reached.
    pub fn register(&self) -> Result<(), SessionError> {
        // CAS loop to atomically increment only when below the cap.
        loop {
            let current = self.active_count.load(Ordering::Relaxed);
            if current >= self.max_sessions {
                return Err(SessionError::InvalidStateTransition {
                    from: SessionState::Active,
                    to: SessionState::Active,
                });
            }
            if self
                .active_count
                .compare_exchange_weak(
                    current,
                    current + 1,
                    Ordering::AcqRel,
                    Ordering::Relaxed,
                )
                .is_ok()
            {
                return Ok(());
            }
        }
    }

    /// Unregister a session.
    pub fn unregister(&self) {
        // Saturating decrement to avoid underflow.
        let _ = self.active_count.fetch_update(
            Ordering::AcqRel,
            Ordering::Relaxed,
            |v| Some(v.saturating_sub(1)),
        );
    }

    /// Get active session count.
    pub fn active_count(&self) -> usize {
        self.active_count.load(Ordering::Relaxed)
    }

    /// Get maximum sessions.
    pub fn max_sessions(&self) -> usize {
        self.max_sessions
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_session_config_default() {
        let config = SessionConfig::default();
        assert!(config.ipc_prefix.starts_with("/dev/shm/szca_"));
        assert_eq!(config.sample_rate, 16000);
        assert_eq!(config.chunk_duration_ms, 20);
    }

    #[test]
    fn test_session_new() {
        let config = SessionConfig::default();
        let session = Session::new(config);
        assert_eq!(session.state(), SessionState::Created);
        assert!(!session.is_cancelled());
        assert!(!session.is_tts_playing());
    }

    #[test]
    fn test_session_has_unique_id() {
        let config = SessionConfig::default();
        let s1 = Session::new(config.clone());
        let s2 = Session::new(config);
        assert_ne!(s1.id(), s2.id());
    }

    #[test]
    fn test_session_activate() {
        let config = SessionConfig::default();
        let mut session = Session::new(config);
        assert!(session.activate().is_ok());
        assert_eq!(session.state(), SessionState::Active);
    }

    #[test]
    fn test_session_activate_invalid_transition() {
        let config = SessionConfig::default();
        let mut session = Session::new(config);
        session.activate().unwrap();
        assert_eq!(
            session.activate(),
            Err(SessionError::InvalidStateTransition {
                from: SessionState::Active,
                to: SessionState::Active
            })
        );
    }

    #[test]
    fn test_session_pause() {
        let config = SessionConfig::default();
        let mut session = Session::new(config);
        session.activate().unwrap();
        assert!(session.pause().is_ok());
        assert_eq!(session.state(), SessionState::Paused);
    }

    #[test]
    fn test_session_pause_invalid() {
        let config = SessionConfig::default();
        let mut session = Session::new(config);
        assert_eq!(
            session.pause(),
            Err(SessionError::InvalidStateTransition {
                from: SessionState::Created,
                to: SessionState::Paused
            })
        );
    }

    #[test]
    fn test_session_resume() {
        let config = SessionConfig::default();
        let mut session = Session::new(config);
        session.activate().unwrap();
        session.pause().unwrap();
        assert!(session.resume().is_ok());
        assert_eq!(session.state(), SessionState::Active);
    }

    #[test]
    fn test_session_resume_invalid() {
        let config = SessionConfig::default();
        let mut session = Session::new(config);
        session.activate().unwrap();
        assert_eq!(
            session.resume(),
            Err(SessionError::InvalidStateTransition {
                from: SessionState::Active,
                to: SessionState::Active
            })
        );
    }

    #[test]
    fn test_session_end() {
        let config = SessionConfig::default();
        let mut session = Session::new(config);
        session.end();
        assert_eq!(session.state(), SessionState::Ended);
        assert!(session.is_cancelled());
    }

    #[test]
    fn test_session_barge_in() {
        let config = SessionConfig::default();
        let mut session = Session::new(config);
        session.set_tts_playing(true);
        session.barge_in();
        assert!(session.is_cancelled());
        assert!(!session.is_tts_playing());
    }

    #[test]
    fn test_session_reset_barge_in() {
        let config = SessionConfig::default();
        let mut session = Session::new(config);
        session.barge_in();
        assert!(session.is_cancelled());
        session.reset_barge_in();
        assert!(!session.is_cancelled());
    }

    #[test]
    fn test_session_cancel_flag() {
        let config = SessionConfig::default();
        let mut session = Session::new(config);
        let flag = session.cancel_flag();

        session.barge_in();
        assert!(flag.load(Ordering::Relaxed));

        session.reset_barge_in();
        assert!(!flag.load(Ordering::Relaxed));
    }

    #[test]
    fn test_session_stats() {
        let config = SessionConfig::default();
        let mut session = Session::new(config);

        session.record_bytes_in(1000);
        session.record_bytes_out(2000);
        session.record_stt_partial();
        session.record_stt_partial();
        session.record_stt_final();
        session.record_llm_token();
        session.record_llm_token();
        session.record_llm_token();
        session.record_tts_chunk();
        session.record_latency(42.5);

        let stats = session.stats();
        assert_eq!(stats.bytes_in, 1000);
        assert_eq!(stats.bytes_out, 2000);
        assert_eq!(stats.stt_partials, 2);
        assert_eq!(stats.stt_finals, 1);
        assert_eq!(stats.llm_tokens, 3);
        assert_eq!(stats.tts_chunks, 1);
        assert_eq!(stats.last_latency_ms, 42.5);
    }

    #[test]
    fn test_session_tts_playing() {
        let config = SessionConfig::default();
        let mut session = Session::new(config);

        assert!(!session.is_tts_playing());
        session.set_tts_playing(true);
        assert!(session.is_tts_playing());
        session.set_tts_playing(false);
        assert!(!session.is_tts_playing());
    }

    #[test]
    fn test_session_error_display() {
        let err = SessionError::InvalidStateTransition {
            from: SessionState::Created,
            to: SessionState::Paused,
        };
        let msg = format!("{}", err);
        assert!(msg.contains("Created"));
        assert!(msg.contains("Paused"));
    }

    #[test]
    fn test_session_manager_new() {
        let manager = SessionManager::new(100);
        assert_eq!(manager.active_count(), 0);
        assert_eq!(manager.max_sessions(), 100);
        assert!(manager.can_create());
    }

    #[test]
    fn test_session_manager_register() {
        let manager = SessionManager::new(2);
        assert!(manager.register().is_ok());
        assert_eq!(manager.active_count(), 1);
        assert!(manager.register().is_ok());
        assert_eq!(manager.active_count(), 2);
        assert!(!manager.can_create());
    }

    #[test]
    fn test_session_manager_unregister() {
        let manager = SessionManager::new(2);
        manager.register().unwrap();
        manager.register().unwrap();
        manager.unregister();
        assert_eq!(manager.active_count(), 1);
        assert!(manager.can_create());
    }

    #[test]
    fn test_session_manager_unregister_empty() {
        let manager = SessionManager::new(2);
        manager.unregister(); // Should not panic
        assert_eq!(manager.active_count(), 0);
    }

    #[test]
    fn test_session_manager_full() {
        let manager = SessionManager::new(1);
        manager.register().unwrap();
        assert_eq!(
            manager.register(),
            Err(SessionError::InvalidStateTransition {
                from: SessionState::Active,
                to: SessionState::Active
            })
        );
    }

    #[test]
    fn test_session_config_accessor() {
        let config = SessionConfig::default();
        let session = Session::new(config);
        assert_eq!(session.config().sample_rate, 16000);
    }
}
