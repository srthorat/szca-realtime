# Changelog

All notable changes to SZCA will be documented in this file.

Format based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [5.1.0] - 2026-07-26

### Added

#### Gateway (Rust)
- `<EOU>` / Streaming STT early turn-taking in `rt_session.rs` for sub-100ms turn latency; live `TranscriptDelta` events over WebSocket during audio stream accumulation.
- `SttChunkResult` and `push_chunk` / `reset_stream` trait methods on `SttStage`.

#### Tooling & Deployment
- Supply-Chain Hardening: Pinned base image immutable `@sha256:...` digests across `Dockerfile` (`rust:1.92-slim-bookworm`, `ubuntu:22.04`, `nvidia/cuda:12.4.0`) and `Dockerfile.dev`.
- `download_models.sh --with-streaming` CLI flag and `WITH_STREAMING=1` environment variable to gate ~400 MB streaming STT model downloads.
- Pinned ONNX Runtime 1.22.0 in production `Dockerfile` with SHA-256 integrity verification (matching `Dockerfile.dev`).
- Added `/opt/onnxruntime` header and library paths to `szca_onnx_engine/CMakeLists.txt`.

---

## [5.0.0] - 2026-07-22

### Added

#### Architecture
- Pure streaming architecture (no batching)
- 4-API design: Unified Voice, STT, LLM, TTS
- Hardware execution provider abstraction (NVIDIA, AMD, Intel, Apple, CPU)
- Dynamic ONNX Runtime EP binding

#### Gateway (Rust)
- Lock-free SPSC ring buffer with cache-line alignment
- Binary wire protocol (16kHz PCM)
- DeepFilterNet3 noise suppression (SIMD)
- Silero VAD v5 speech detection
- Atomic barge-in interrupts
- POSIX shared memory IPC
- Session management with state machine
- WebSocket server (Axum + Tokio)
- HTTP SSE endpoints for STT, LLM, TTS
- Health check endpoint (`/health`)
- Metrics endpoint (`/metrics`)
- Graceful shutdown (SIGTERM handler)
- Request timeout (30s)
- Structured logging (tracing)

#### Engine (C++)
- ONNX Runtime integration (CUDA, ROCm, OpenVINO, CoreML, CPU)
- Parakeet TDT 0.6B V3 STT (FP16)
- Hermes-3-Llama-3.2-3B INT8 LLM
- Kokoro-82M TTS (8 languages, 54 voices)
- SoXR resampler (24kHz → 16kHz)
- POSIX shared memory IPC
- Session management

#### CPU Deployment
- CPU-only mode (no GPU required)
- HTTP server for API endpoints
- Docker support
- Works on x86_64 and arm64 (Apple Silicon)

#### Testing
- 341 total tests
- 119 Rust gateway unit tests
- 74 C++ engine unit tests
- 15 integration tests
- 12 E2E tests
- 10 performance benchmarks
- 18 security tests
- 83 LLM correctness tests
- 40 LLM advanced tests

#### Deployment
- Dockerfile for CPU and GPU modes
- docker-compose.yml
- Model download script
- GitHub Actions CI/CD pipeline
- k6 load testing scripts

#### Documentation
- Architecture document (v5.0.0)
- README (753 lines, 16 sections)
- Test plan (341 tests)
- TODO list (18 items)
- CHANGELOG

### Changed
- Upgraded from Parakeet TDT 1.1B to 0.6B V3 (smaller, faster)
- Upgraded latency budget from 96.6ms to 51-56ms
- Updated cost model to $1,700/mo (on-premise)

### Fixed
- Graceful error handling (no `unwrap()` in production)
- Structured logging (replaced `eprintln!` with `tracing`)
- Request timeout (prevents resource exhaustion)
- Graceful shutdown (SIGTERM handler)

---

## [4.0.0] - 2026-07-22

### Added
- Dynamic hardware EP abstraction
- Unified C++ ONNX inference engine
- Multilingual capability matrix
- Kyutai Moshi comparison
- Two-artifact build (28MB gateway + C++ engine)

### Changed
- Updated latency budget to 80.1ms
- Updated cost model to $14,110/mo

---

## [3.0.0] - 2026-07-22

### Added
- Pure streaming architecture (no batching)
- AWS deployment configuration
- Token throughput scaling
- GPU benchmark matrix

---

## [2.0.0] - 2026-07-22

### Added
- CPU Pure C++ architecture analysis
- Hardware-agnostic design

---

## [1.0.0] - 2026-07-22

### Added
- Initial architecture document
- GPU NVLink cluster design
- TensorRT-LLM integration
- EAGLE-2 speculative decoding

---

*Format: https://keepachangelog.com/en/1.0.0/*
