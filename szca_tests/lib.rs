//! SZCA reference test crate (`szca_tests`)
//!
//! **Scope:** lightweight mock pipelines and reference logic — this crate does
//! **not** depend on `szca_media_gateway` and does not run real ONNX inference.
//!
//! | Area | What runs here | Real gateway coverage |
//! |------|----------------|------------------------|
//! | Integration | `simulate_pipeline` stubs | `szca_media_gateway/tests/e2e_pipeline.rs` |
//! | E2E journeys | Mock user flows | `rust-tests` job (`cargo test` in gateway) |
//! | Security | Reference `validate_auth` helpers | Gateway `require_auth` in `api_routes.rs` |
//! | Performance | In-test timing of mocks | Real-weights tests (opt-in, local/CI manual) |
//! | Metrics | LLM metric calculators | Gateway unit tests + Prometheus endpoint |
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
