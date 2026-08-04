/// WebSocket session admission control (concurrent connection cap).

use std::sync::atomic::{AtomicUsize, Ordering};

/// Admission limit reached.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AdmissionError;

impl std::fmt::Display for AdmissionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "session limit reached")
    }
}

impl std::error::Error for AdmissionError {}

/// Tracks active WebSocket sessions for admission control.
///
/// Thread-safe via an atomic counter, so it can be shared behind an `Arc` across
/// connection handlers.
pub struct SessionManager {
    max_sessions: usize,
    active_count: AtomicUsize,
}

impl SessionManager {
    /// Create a manager with the given concurrent-session cap.
    pub fn new(max_sessions: usize) -> Self {
        Self {
            max_sessions,
            active_count: AtomicUsize::new(0),
        }
    }

    /// Whether a new session can be accepted.
    pub fn can_create(&self) -> bool {
        self.active_count.load(Ordering::Relaxed) < self.max_sessions
    }

    /// Reserve a slot for a new session.
    pub fn register(&self) -> Result<(), AdmissionError> {
        loop {
            let current = self.active_count.load(Ordering::Relaxed);
            if current >= self.max_sessions {
                return Err(AdmissionError);
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

    /// Release a slot when a session ends.
    pub fn unregister(&self) {
        let _ = self.active_count.fetch_update(
            Ordering::AcqRel,
            Ordering::Relaxed,
            |v| Some(v.saturating_sub(1)),
        );
    }

    /// Current active session count.
    pub fn active_count(&self) -> usize {
        self.active_count.load(Ordering::Relaxed)
    }

    /// Configured concurrent-session cap.
    pub fn max_sessions(&self) -> usize {
        self.max_sessions
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn session_manager_new() {
        let manager = SessionManager::new(100);
        assert_eq!(manager.active_count(), 0);
        assert_eq!(manager.max_sessions(), 100);
        assert!(manager.can_create());
    }

    #[test]
    fn session_manager_register() {
        let manager = SessionManager::new(2);
        assert!(manager.register().is_ok());
        assert_eq!(manager.active_count(), 1);
        assert!(manager.register().is_ok());
        assert_eq!(manager.active_count(), 2);
        assert!(!manager.can_create());
    }

    #[test]
    fn session_manager_unregister() {
        let manager = SessionManager::new(2);
        manager.register().unwrap();
        manager.register().unwrap();
        manager.unregister();
        assert_eq!(manager.active_count(), 1);
        assert!(manager.can_create());
    }

    #[test]
    fn session_manager_unregister_empty() {
        let manager = SessionManager::new(2);
        manager.unregister();
        assert_eq!(manager.active_count(), 0);
    }

    #[test]
    fn session_manager_full() {
        let manager = SessionManager::new(1);
        manager.register().unwrap();
        assert_eq!(manager.register(), Err(AdmissionError));
    }
}
