//! SZCA Test Suite
//!
//! Comprehensive tests for the SZCA voice engine:
//! - Integration tests (pipeline, component interaction)
//! - E2E tests (full user journeys)
//! - Performance benchmarks (throughput, latency)
//! - Security tests (auth, injection, rate limiting)
//! - Metrics tests (STT WER, LLM TTFT/TPOT, TTS quality)
//!
//! The module files do not follow Rust's default `mod name -> name.rs`
//! layout, so each declaration uses an explicit `#[path]` attribute that
//! points at the real file on disk.

#[path = "integration/test_pipeline.rs"]
pub mod integration_test_pipeline;

#[path = "e2e/test_end_to_end.rs"]
pub mod e2e_test_end_to_end;

#[path = "performance/test_benchmarks.rs"]
pub mod performance_test_benchmarks;

#[path = "security/test_security.rs"]
pub mod security_test_security;

#[path = "metrics/test_metrics.rs"]
pub mod metrics_test_metrics;

#[path = "metrics/test_llm_comprehensive.rs"]
pub mod metrics_test_llm_comprehensive;

#[path = "metrics/test_llm_advanced.rs"]
pub mod metrics_test_llm_advanced;
