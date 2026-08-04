/// Metrics collection and export for SZCA gateway.
///
/// Provides Prometheus-compatible metrics for monitoring.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

/// Global metrics collector.
pub struct Metrics {
    pub requests_total: AtomicU64,
    pub requests_success: AtomicU64,
    pub requests_failed: AtomicU64,
    pub sessions_active: AtomicU64,
    pub sessions_total: AtomicU64,
    pub bytes_in: AtomicU64,
    pub bytes_out: AtomicU64,
    pub stt_partials: AtomicU64,
    pub stt_finals: AtomicU64,
    pub llm_tokens: AtomicU64,
    pub tts_chunks: AtomicU64,
    start_time: Instant,
}

impl Metrics {
    /// Create a new metrics collector.
    pub fn new() -> Self {
        Self {
            requests_total: AtomicU64::new(0),
            requests_success: AtomicU64::new(0),
            requests_failed: AtomicU64::new(0),
            sessions_active: AtomicU64::new(0),
            sessions_total: AtomicU64::new(0),
            bytes_in: AtomicU64::new(0),
            bytes_out: AtomicU64::new(0),
            stt_partials: AtomicU64::new(0),
            stt_finals: AtomicU64::new(0),
            llm_tokens: AtomicU64::new(0),
            tts_chunks: AtomicU64::new(0),
            start_time: Instant::now(),
        }
    }

    /// Get uptime in seconds.
    pub fn uptime_secs(&self) -> u64 {
        self.start_time.elapsed().as_secs()
    }

    /// Record a request.
    pub fn record_request(&self, success: bool) {
        self.requests_total.fetch_add(1, Ordering::Relaxed);
        if success {
            self.requests_success.fetch_add(1, Ordering::Relaxed);
        } else {
            self.requests_failed.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Record a session start.
    pub fn record_session_start(&self) {
        self.sessions_active.fetch_add(1, Ordering::Relaxed);
        self.sessions_total.fetch_add(1, Ordering::Relaxed);
    }

    /// Record a session end.
    pub fn record_session_end(&self) {
        // Saturating decrement to avoid underflow if this is called more times
        // than record_session_start.
        let _ = self.sessions_active.fetch_update(
            Ordering::Relaxed,
            Ordering::Relaxed,
            |v| Some(v.saturating_sub(1)),
        );
    }

    /// Record bytes received.
    pub fn record_bytes_in(&self, bytes: u64) {
        self.bytes_in.fetch_add(bytes, Ordering::Relaxed);
    }

    /// Record bytes sent.
    pub fn record_bytes_out(&self, bytes: u64) {
        self.bytes_out.fetch_add(bytes, Ordering::Relaxed);
    }

    /// Record STT partial result.
    pub fn record_stt_partial(&self) {
        self.stt_partials.fetch_add(1, Ordering::Relaxed);
    }

    /// Record STT final result.
    pub fn record_stt_final(&self) {
        self.stt_finals.fetch_add(1, Ordering::Relaxed);
    }

    /// Record LLM token generated.
    pub fn record_llm_token(&self) {
        self.llm_tokens.fetch_add(1, Ordering::Relaxed);
    }

    /// Record TTS chunk generated.
    pub fn record_tts_chunk(&self) {
        self.tts_chunks.fetch_add(1, Ordering::Relaxed);
    }

    /// Export metrics as Prometheus text format.
    pub fn export_prometheus(&self) -> String {
        format!(
            "# HELP szca_requests_total Total number of requests\n\
             # TYPE szca_requests_total counter\n\
             szca_requests_total {}\n\
             \n\
             # HELP szca_requests_success Successful requests\n\
             # TYPE szca_requests_success counter\n\
             szca_requests_success {}\n\
             \n\
             # HELP szca_requests_failed Failed requests\n\
             # TYPE szca_requests_failed counter\n\
             szca_requests_failed {}\n\
             \n\
             # HELP szca_sessions_active Active sessions\n\
             # TYPE szca_sessions_active gauge\n\
             szca_sessions_active {}\n\
             \n\
             # HELP szca_sessions_total Total sessions\n\
             # TYPE szca_sessions_total counter\n\
             szca_sessions_total {}\n\
             \n\
             # HELP szca_bytes_in Total bytes received\n\
             # TYPE szca_bytes_in counter\n\
             szca_bytes_in {}\n\
             \n\
             # HELP szca_bytes_out Total bytes sent\n\
             # TYPE szca_bytes_out counter\n\
             szca_bytes_out {}\n\
             \n\
             # HELP szca_stt_partials STT partial results\n\
             # TYPE szca_stt_partials counter\n\
             szca_stt_partials {}\n\
             \n\
             # HELP szca_stt_finals STT final results\n\
             # TYPE szca_stt_finals counter\n\
             szca_stt_finals {}\n\
             \n\
             # HELP szca_llm_tokens LLM tokens generated\n\
             # TYPE szca_llm_tokens counter\n\
             szca_llm_tokens {}\n\
             \n\
             # HELP szca_tts_chunks TTS audio chunks generated\n\
             # TYPE szca_tts_chunks counter\n\
             szca_tts_chunks {}\n\
             \n\
             # HELP szca_uptime_seconds Server uptime in seconds\n\
             # TYPE szca_uptime_seconds gauge\n\
             szca_uptime_seconds {}\n",
            self.requests_total.load(Ordering::Relaxed),
            self.requests_success.load(Ordering::Relaxed),
            self.requests_failed.load(Ordering::Relaxed),
            self.sessions_active.load(Ordering::Relaxed),
            self.sessions_total.load(Ordering::Relaxed),
            self.bytes_in.load(Ordering::Relaxed),
            self.bytes_out.load(Ordering::Relaxed),
            self.stt_partials.load(Ordering::Relaxed),
            self.stt_finals.load(Ordering::Relaxed),
            self.llm_tokens.load(Ordering::Relaxed),
            self.tts_chunks.load(Ordering::Relaxed),
            self.uptime_secs(),
        )
    }

    /// Export metrics including stage pool health and latency quantiles as Prometheus format.
    pub fn export_prometheus_with_pools(&self, pools: Option<&crate::stage_pools::StagePools>) -> String {
        let mut out = self.export_prometheus();

        if let Some(pools) = pools {
            out.push_str("\n# HELP szca_pool_queue_depth Current queue depth per stage pool\n");
            out.push_str("# TYPE szca_pool_queue_depth gauge\n");
            if let Some(ref p) = pools.stt {
                out.push_str(&format!("szca_pool_queue_depth{{stage=\"stt\"}} {}\n", p.queue_depth()));
            } else if let Some(ref p) = pools.streaming_stt {
                out.push_str(&format!("szca_pool_queue_depth{{stage=\"stt\"}} {}\n", p.in_use()));
            }
            if let Some(ref p) = pools.llm {
                out.push_str(&format!("szca_pool_queue_depth{{stage=\"llm\"}} {}\n", p.queue_depth()));
            }
            if let Some(ref p) = pools.tts {
                out.push_str(&format!("szca_pool_queue_depth{{stage=\"tts\"}} {}\n", p.queue_depth()));
            }

            out.push_str("\n# HELP szca_pool_replicas Number of active replicas per stage pool\n");
            out.push_str("# TYPE szca_pool_replicas gauge\n");
            if let Some(ref p) = pools.stt {
                out.push_str(&format!("szca_pool_replicas{{stage=\"stt\"}} {}\n", p.replica_count()));
            } else if let Some(ref p) = pools.streaming_stt {
                out.push_str(&format!("szca_pool_replicas{{stage=\"stt\"}} {}\n", p.replica_count()));
            }
            if let Some(ref p) = pools.llm {
                out.push_str(&format!("szca_pool_replicas{{stage=\"llm\"}} {}\n", p.replica_count()));
            }
            if let Some(ref p) = pools.tts {
                out.push_str(&format!("szca_pool_replicas{{stage=\"tts\"}} {}\n", p.replica_count()));
            }

            out.push_str("\n# HELP szca_pool_latency_milliseconds Execution latency quantiles per stage pool\n");
            out.push_str("# TYPE szca_pool_latency_milliseconds summary\n");

            let stages = [
                ("stt", pools.stt.as_ref().and_then(|p| p.latency_snapshot())),
                ("llm", pools.llm.as_ref().and_then(|p| p.latency_snapshot())),
                ("tts", pools.tts.as_ref().and_then(|p| p.latency_snapshot())),
            ];

            for (stage_name, snap_opt) in stages {
                if let Some(snap) = snap_opt {
                    out.push_str(&format!("szca_pool_latency_milliseconds{{stage=\"{stage_name}\",quantile=\"0.5\"}} {}\n", snap.p50_ms));
                    out.push_str(&format!("szca_pool_latency_milliseconds{{stage=\"{stage_name}\",quantile=\"0.9\"}} {}\n", snap.p90_ms));
                    out.push_str(&format!("szca_pool_latency_milliseconds{{stage=\"{stage_name}\",quantile=\"0.95\"}} {}\n", snap.p95_ms));
                    out.push_str(&format!("szca_pool_latency_milliseconds{{stage=\"{stage_name}\",quantile=\"0.99\"}} {}\n", snap.p99_ms));
                    out.push_str(&format!("szca_pool_latency_milliseconds_count{{stage=\"{stage_name}\"}} {}\n", snap.count));
                }
            }
        }

        out
    }
}

impl Default for Metrics {
    fn default() -> Self {
        Self::new()
    }
}

/// Shared metrics state.
pub type SharedMetrics = Arc<Metrics>;

/// Create shared metrics.
pub fn create_shared_metrics() -> SharedMetrics {
    Arc::new(Metrics::new())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metrics_new() {
        let metrics = Metrics::new();
        assert_eq!(metrics.requests_total.load(Ordering::Relaxed), 0);
        assert_eq!(metrics.sessions_active.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn test_metrics_record_request() {
        let metrics = Metrics::new();
        metrics.record_request(true);
        metrics.record_request(true);
        metrics.record_request(false);

        assert_eq!(metrics.requests_total.load(Ordering::Relaxed), 3);
        assert_eq!(metrics.requests_success.load(Ordering::Relaxed), 2);
        assert_eq!(metrics.requests_failed.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn test_metrics_sessions() {
        let metrics = Metrics::new();
        metrics.record_session_start();
        metrics.record_session_start();
        assert_eq!(metrics.sessions_active.load(Ordering::Relaxed), 2);
        assert_eq!(metrics.sessions_total.load(Ordering::Relaxed), 2);

        metrics.record_session_end();
        assert_eq!(metrics.sessions_active.load(Ordering::Relaxed), 1);
        assert_eq!(metrics.sessions_total.load(Ordering::Relaxed), 2);
    }

    #[test]
    fn test_metrics_bytes() {
        let metrics = Metrics::new();
        metrics.record_bytes_in(1000);
        metrics.record_bytes_out(2000);

        assert_eq!(metrics.bytes_in.load(Ordering::Relaxed), 1000);
        assert_eq!(metrics.bytes_out.load(Ordering::Relaxed), 2000);
    }

    #[test]
    fn test_metrics_stt() {
        let metrics = Metrics::new();
        metrics.record_stt_partial();
        metrics.record_stt_partial();
        metrics.record_stt_final();

        assert_eq!(metrics.stt_partials.load(Ordering::Relaxed), 2);
        assert_eq!(metrics.stt_finals.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn test_metrics_llm() {
        let metrics = Metrics::new();
        for _ in 0..10 {
            metrics.record_llm_token();
        }
        assert_eq!(metrics.llm_tokens.load(Ordering::Relaxed), 10);
    }

    #[test]
    fn test_metrics_tts() {
        let metrics = Metrics::new();
        for _ in 0..5 {
            metrics.record_tts_chunk();
        }
        assert_eq!(metrics.tts_chunks.load(Ordering::Relaxed), 5);
    }

    #[test]
    fn test_metrics_underflow_saturates() {
        // record_session_end without a matching start must not underflow.
        let metrics = Metrics::new();
        metrics.record_session_end();
        assert_eq!(metrics.sessions_active.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn test_metrics_uptime() {
        use std::time::Duration;
        let metrics = Metrics::new();
        std::thread::sleep(Duration::from_millis(1100));
        // After sleeping >1s, uptime must have advanced to at least 1 second.
        assert!(metrics.uptime_secs() >= 1);
    }

    #[test]
    fn test_metrics_prometheus_export() {
        let metrics = Metrics::new();
        metrics.record_request(true);
        metrics.record_llm_token();

        let prometheus = metrics.export_prometheus();
        assert!(prometheus.contains("szca_requests_total 1"));
        assert!(prometheus.contains("szca_llm_tokens 1"));
        assert!(prometheus.contains("# TYPE szca_requests_total counter"));
    }

    #[test]
    fn test_metrics_prometheus_export_with_pools() {
        let metrics = Metrics::new();
        let prometheus_no_pools = metrics.export_prometheus_with_pools(None);
        assert!(prometheus_no_pools.contains("szca_requests_total 0"));

        let pools = crate::stage_pools::StagePools {
            stt: None,
            streaming_stt: None,
            llm: None,
            tts: None,
        };
        let prometheus_pools = metrics.export_prometheus_with_pools(Some(&pools));
        assert!(prometheus_pools.contains("# HELP szca_pool_queue_depth"));
        assert!(prometheus_pools.contains("# HELP szca_pool_replicas"));
    }
}
