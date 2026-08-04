# SZCA Realtime Voice Engine — Single Source of Truth

**Project:** SRAM-Mesh Zero-Copy Architecture (SZCA) Real-Time Voice Engine
**Version:** 5.0.0
**Date:** July 2026
**Status:** Phases 1–6 Complete · Phase 7 Backlog

---

> **What this is:** A production Rust realtime voice engine (`szca_media_gateway`):
> a bidirectional-streaming alternative to OpenAI Realtime / Gemini Live,
> cascading **STT → LLM → TTS** with server-side VAD, barge-in, and both
> OpenAI Realtime and Gemini Live wire dialects. Behind a concurrent,
> composable StagePool architecture with config-driven LLM backend selection
> (in-process ONNX or external vLLM/TGI).

---

## Table of Contents

1. [Architecture](#1-architecture)
2. [Model Stack & Latency](#2-model-stack--latency)
3. [What's Built (Phases 0–6)](#3-whats-built)
4. [Deployment Profiles](#4-deployment-profiles)
5. [Environment Variables](#5-environment-variables)
6. [API Reference](#6-api-reference)
7. [Hardware Sizing](#7-hardware-sizing)
8. [Cost Analysis](#8-cost-analysis)
9. [Dev Setup](#9-dev-setup)
10. [Prod Setup](#10-prod-setup)
11. [Testing](#11-testing)
12. [CI/CD](#12-cicd)
13. [Runtime Abstraction](#13-runtime-abstraction-dont-lock-into-vllm)
14. [Key Decisions](#14-key-decisions-archived)
15. [What's Left (Backlog)](#15-whats-left-backlog)
16. [Dependency Audit](#16-dependency-audit--streaming-stt-parakeet-eou-120m)
17. [Source Map](#17-source-map)
18. [Glossary](#18-glossary)

---

## 1. Architecture

### Design Principles

| Principle | Implementation |
|-----------|---------------|
| Zero-Copy IPC | POSIX Shared Memory (`/dev/shm`) — no TCP loopback, no serialization |
| Lock-Free Hot Path | Pre-allocated SPSC ring buffers, `AtomicBool` cancellation, 0-byte alloc on audio path |
| Pure Streaming | Each token processed immediately — no accumulation, no batch wait |
| Binary Wire Protocol | Raw PCM WebSocket frames — zero Base64, zero JSON for audio |
| Hardware-Agnostic | ONNX Runtime EP abstraction — NVIDIA, AMD, Intel, Apple, CPU |

### Locked Architecture

Ultrascalable voice = **our Rust engine owns STT/TTS/VAD/orchestration; vLLM (or TGI) owns LLM token generation on your GPUs.** We do NOT build our own LLM batching engine — that is re-implementing vLLM on a runtime (ORT) without the custom CUDA kernels, months of work, and still slower.

For **voice**, perceived "big-player" quality = low time-to-first-audio + full streaming + fast barge-in (all largely built), NOT raw tok/s. ~30–50 tok/s already outpaces human listening.

```
OUR RUST ENGINE                              vLLM / TGI on your GPUs
────────────────                             ───────────────────────
• STT pool (Parakeet)  in-process ONNX       • Hermes/Llama generation
• TTS pool (Kokoro)    in-process ONNX       • continuous batching
• VAD / barge-in / turn-taking               • paged KV-cache
• streaming orchestration, dual dialect      • tensor-parallel, 1000s streams
• LLM stage = streaming CLIENT ───────────►  OpenAI-compatible API (stream=true)
```

**NOT our scope** (provided to the engine as requests when needed): flow engine / deterministic state machine, RAG pipeline, guardrail/validation layer. The engine exposes seams these plug into (in-process pool API + WS/HTTP endpoints).

### Target Architecture

```
                        ┌──────────────────────────────────────┐
   WS realtime  ─┐      │            StagePools (shared)         │
   sessions      ├────► │  ┌─ SttPool  : N_stt replicas + queue  │
   HTTP /v1/*   ─┘      │  ├─ LlmPool  : N_llm replicas + queue  │
   sessions             │  └─ TtsPool  : N_tts replicas + queue  │
                        └──────────────────────────────────────┘
                          each replica = 1 model instance on 1 OS thread
                          each pool    = MPSC job queue drained by replicas
```

### End-to-End Pipeline

```
Customer Gateway
     │ 16kHz PCM In (binary WebSocket)
     ▼
┌─────────────────────────────────────────────────────┐
│  LAYER 1: RUST MEDIA GATEWAY (szca_media_gateway)  │
│  • DeepFilterNet3 SIMD (noise suppression, <1.5ms) │
│  • Silero VAD v5 ONNX (speech detection, <0.5ms)  │
│  • Atomic Interrupt Controller (barge-in, <0.01ms) │
└──────────────────────┬──────────────────────────────┘
                       │ Zero-Copy SHM Ring Buffer
                       ▼
┌─────────────────────────────────────────────────────┐
│  LAYER 2: INFERENCE (in-process ORT + vLLM)        │
│  STT ──► LLM ──► TTS ──► Resampler (24k → 16k)    │
│  Parakeet* Qwen/vLLM  Kokoro   SoXR               │
│  TDT 0.6B  1.5B/8B    82M      0.5ms              │
│  *or EOU 120M / Zipformer via STT_BACKEND          │
│  ~22ms     ~1.5ms/tok ~10ms                        │
└──────────────────────┬──────────────────────────────┘
                       │ 16kHz PCM Out (binary WebSocket)
                       ▼
                   Customer Speaker
```

### Fungibility

`STT_REPLICAS`, `LLM_REPLICAS`, `TTS_REPLICAS` are independent env config. A stage with `N = 0` is disabled. So:

| Deployment | Config |
|---|---|
| Full S2S ×K | `STT=K LLM=K TTS=K` |
| STT-only ×100 | `STT=100 LLM=0 TTS=0` |
| 100 STT + 200 (LLM+TTS) | `STT=100 LLM=200 TTS=200` |

---

## 2. Model Stack & Latency

### Models

| Stage | Model | Precision | Size | License |
|-------|-------|-----------|------|---------|
| Noise suppression | DeepFilterNet3 | FP32 ONNX | ~10 MB | Apache-2.0 |
| VAD | Silero VAD v5 (16 kHz) | FP32 ONNX | ~2 MB | MIT |
| STT | Parakeet TDT 0.6B V3 | INT8 ONNX | ~400 MB | CC-BY-4.0 |
| STT (streaming) | Parakeet EOU 120M (cache-aware) | FP16 ONNX | ~236 MB | CC-BY-4.0 |
| STT (streaming) | Sherpa Zipformer 19-layer (cache-aware) | FP32 ONNX | ~156 MB | Apache-2.0 |
| LLM (dev) | Hermes-3-Llama-3.2-3B Instruct | FP32 ONNX (external data) | ~14.4 GB on disk | Llama 3.2 Community |
| LLM (prod) | Hermes-3-Llama-3.1-8B Instruct | FP8 (vLLM) | ~8 GB | Llama 3.1 Community |
| TTS | Kokoro-82M v1.0 | FP16 ONNX | ~170 MB | Apache-2.0 |
| TTS G2P | Misaki (pure-Rust port) | — | — | MIT |

All models commercially usable (MIT / Apache 2.0 / CC-BY-4.0).

### Latency Budget — DEV (Qwen2.5-1.5B INT8 ONNX, CPU)

| Stage | Technology | Latency |
|-------|-----------|---------|
| Noise Filtering | DeepFilterNet3 SIMD | ~1.5 ms |
| Speech Detection | Silero VAD v5 ONNX | ~0.5 ms |
| STT | Parakeet TDT 0.6B V3 INT8 | ~22 ms |
| Shared Memory IPC | POSIX Lock-Free SHM | ~0.1 ms |
| LLM TTFT | Qwen2.5-1.5B INT8 ONNX (CPU) | ~100–500 ms (estimated, M-series) |
| LLM Decode | — | ~10–20 tok/s (estimated, INT8 CPU) |
| TTS First Chunk | Kokoro-82M ONNX | ~10 ms |
| Resampling | SoXR 24k → 16k | ~0.5 ms |
| WebSocket Egress | Rust Axum Binary Writer | ~1 ms |
| **TOTAL (First Audio)** | | **~150–550 ms** |

> **Dev is a correctness harness, not a latency target.** With Qwen2.5-1.5B INT8
> as the default dev LLM, the pipeline is already at interactive speed (~10-20
> tok/s). Hermes-3-Llama-3.2-3B FP32 (~2-10 s TTFT, ~1-3 tok/s) is available via
> `LLM_MODEL=hermes3` for model-family parity testing. Everything except the
> LLM row is already at prod speed, so dev exercises the same code path, the same
> event contract, and the same model family — just slowly.
>
> If you need dev to feel interactive, either (a) keep `LLM_MAX_NEW_TOKENS` small
> (default 96 in `env.dev.example`), or (b) point `LLM_BACKEND=vllm` at a shared
> GPU box — the gateway code is identical either way.

### Latency Budget — PROD (Hermes-3-8B vLLM, A100 GPU)

| Stage | Technology | Latency |
|-------|-----------|---------|
| Noise Filtering | DeepFilterNet3 SIMD | ~1.5 ms |
| Speech Detection | Silero VAD v5 ONNX | ~0.5 ms |
| STT | Parakeet TDT 0.6B V3 INT8 | ~22 ms |
| Shared Memory IPC | POSIX Lock-Free SHM | ~0.1 ms |
| LLM TTFT | Hermes-3-8B FP8 vLLM (A100) | ~50–100 ms |
| LLM Decode | — | ~15–30 ms/token (GPU) |
| TTS First Chunk | Kokoro-82M ONNX | ~10 ms |
| Resampling | SoXR 24k → 16k | ~0.5 ms |
| WebSocket Egress | Rust Axum Binary Writer | ~1 ms |
| **TOTAL (First Audio)** | | **~85–135 ms** |

### Streaming Interface — Per-Stage Reality

| Stage | Input | Output | Interim | Final |
|-------|-------|--------|---------|-------|
| STT | batch utterance (post-VAD)¹ | streaming | `TranscriptDelta` (real) | `TranscriptDone` |
| LLM | text | streaming | `TextDelta` (true token stream) | `TextDone` |
| TTS | text (sentence-chunked) | streaming | `AudioDelta` per sentence | `AudioDone` |

¹ With `STT_BACKEND=parakeet` (default), STT input is the whole post-VAD utterance. `STT_BACKEND=streaming` or `zipformer` selects a cache-aware streaming encoder that consumes audio incrementally in 1.28 s (EOU) or 1.41 s (Zipformer) chunks — cost is linear in stream length, not quadratic. The streaming stage emits `<EOU>` (EOU model) or word-boundary partials (Zipformer). `rt_session` does not yet use the streaming path for turn-end; that is the follow-up.

---

## 3. What's Built

### Transport & Session (Phase 0)

- WebSocket transport (axum 0.7) with dual-dialect adapters (OpenAI Realtime + Gemini Live)
- Neutral event model (`ClientCommand` / `ServerEvent`) — dialect-agnostic session loop
- Server-side VAD turn detection (Silero VAD v5, RMS fallback)
- Barge-in via `AtomicBool` cancel flag; per-turn work in `spawn_blocking`
- `Pipeline` trait seam (`SttStage` / `LlmStage` / `TtsStage`)

### STT — Parakeet TDT 0.6B V3 (INT8) — `rt_stt.rs`

- Three chained ONNX graphs: `nemo128.onnx` → `encoder.int8.onnx` → `decoder_joint.int8.onnx`
- TDT greedy decode: token + duration per step, blank-aware time advancement
- SentencePiece detokenization (8193 entries, `▁` → space)
- Real interim transcripts at each word boundary via `TranscriptDelta`

### LLM — Hermes-3-Llama-3.2-3B (ONNX, dev) / Hermes-3-Llama-3.1-8B (vLLM, prod)

- KV-cache autoregressive decode (3 fixed + 2×N-layer past-KV tensors per forward)
- Greedy argmax + repetition penalty (matches Python oracle)
- True token streaming with UTF-8 `\u{FFFD}` holdback so partial multi-byte never leaks
- `VllmClient` (`rt_llm_client.rs`): streaming HTTP client to `/v1/chat/completions` SSE
- **Model-family agnostic** — nothing about the checkpoint is hard-coded:
  - KV geometry (layers / KV-heads / head-dim) from `config.json`
  - Chat template from the checkpoint's own Jinja `chat_template` in
    `tokenizer_config.json` — ChatML (`<|im_start|>`) vs Llama-3
    (`<|start_header_id|>`); override with `LLM_CHAT_TEMPLATE`
  - Stop tokens = **union** of `generation_config.eos_token_id`,
    `config.eos_token_id`, and the tokenizer's `eos_token`
- Verified checkpoints: Hermes-3-Llama-3.2-3B (FP32, ChatML-on-Llama) and
  Qwen2.5-1.5B-Instruct (int8, ChatML)

### TTS — Kokoro-82M + Misaki G2P — `rt_tts.rs`

- Pure-Rust G2P via `misaki-rs` (no espeak-ng C FFI, no `build.rs`)
- Flow: `text → misaki g2p → phoneme IDs → Kokoro ONNX → 24 kHz → resample → PCM16`
- Voice pack indexed by phoneme length; per-call voice override
- Sentence-chunked streaming: TTS interleaved with LLM, each sentence synthesized as soon as emitted

### StagePool — Concurrent, Composable Serving (Phases 1–6) ✅

**Phase 1:** Generic `StagePool<R: Replica>` core (`stage_pool.rs`)
- Bounded MPMC job queue (crossbeam-channel)
- One blocking thread per replica, streaming deltas via `mpsc::UnboundedSender`
- `Job` type: input + `Arc<AtomicBool>` cancel + result channels
- `try_submit` (non-blocking) + `try_submit_with_cancel` (caller's cancel Arc)
- `queue_depth()` counter, `replica_count()` accessor

**Phase 2a:** STT/TTS/LLM wrapped as pool replicas
- `Replica` impls for `ParakeetStt`, `KokoroTts`, `QwenLlm`
- `stage_pools.rs`: pool adapters + `StagePools::from_env()`
- Models load once at startup, shared across all sessions

**Phase 2b:** vLLM streaming client (`rt_llm_client.rs`)
- `VllmClient` implementing `Replica` trait, streaming HTTP SSE
- Config: `LLM_BACKEND=onnx|vllm`, `LLM_BASE_URL`, `LLM_MODEL`, `LLM_API_KEY`
- `LlmBackend` enum wrapping either in-process ONNX or external vLLM

**Phase 3:** WS sessions rewired onto shared pools
- `run_session` takes `Option<Arc<StagePools>>` — no per-session model copies
- `Mutex<Pipeline>` eliminated — pool adapters are separate structs

**Phase 4:** HTTP endpoints on shared pools
- `/v1/stt/stream`, `/v1/llm/stream`, `/v1/tts/stream` — `spawn_blocking` + SSE
- `/v1/pools` health endpoint, `/metrics` Prometheus export
- Bearer token auth middleware (`SZCA_API_KEY`)

**Phase 5:** Metrics, backpressure, health
- Queue depth counter per pool (`AtomicUsize`)
- 503 + `Retry-After` when pool queue exceeds `POOL_QUEUE_CAP` (64)

**Phase 6:** Production hardening
- Graceful shutdown (SIGTERM + 30s drain)
- Request timeout (tower-http TimeoutLayer, 60s)
- Load testing (k6 HTTP endpoints + WebSocket realtime)
- CI/CD (GitHub Actions: 10 jobs — test, clippy, fmt, audit, Docker, Trivy, gitleaks)

### Dev Environment ✅

- `DEVELOPMENT.md` — developer quickstart (< 5 min to running)
- `Dockerfile.dev` — Rust gateway with cargo-watch hot-reload
- `Dockerfile.llm-dev` — Python dev LLM server (ONNX RT GenAI, CPU)
- `docker-compose.dev.yml` — full dev stack (gateway + optional LLM server)
- `dev_server.py` — FastAPI SSE server, OpenAI-compatible
- `download_model.py` — downloads INT4 ONNX Llama 3.1 8B

---

## 4. Deployment Profiles

The engine runs in two distinct profiles. Same codebase, same API contract — different infra and capacity targets. The `LLM_BACKEND` env var switches between them.

### DEV Profile — ONNX-Only, CPU, Single-User

| Property | Value |
|----------|-------|
| Hardware | Any CPU (MacBook, Linux VM, CI runner) |
| LLM backend | `LLM_BACKEND=onnx` → in-process Qwen2.5-1.5B-Instruct via ORT |
| LLM model | Qwen2.5-1.5B-Instruct **INT8** ONNX (~1.5 GB, self-contained, no external data). Also supports Hermes-3-Llama-3.2-3B FP32 via `LLM_MODEL=hermes3` (~14.4 GB) |
| LLM RAM | ~1.5 GB resident for Qwen2.5; ~15 GB for Hermes-3. Keep `LLM_REPLICAS=1` on a laptop either way |
| Chat template | ChatML, auto-detected from `tokenizer_config.json` (see §14) |
| STT | Parakeet TDT 0.6B int8, in-process ORT |
| TTS | Kokoro-82M + Misaki, in-process ORT |
| Concurrent sessions | 1–2 (single-user correctness) |
| WebSocket | `ws://localhost:3000/v1/realtime?dialect=openai` |
| HTTP | `/v1/stt/stream`, `/v1/llm/stream`, `/v1/tts/stream` |
| Pool sizing | `STT_REPLICAS=1 LLM_REPLICAS=1 TTS_REPLICAS=1` |
| Docker | `docker-compose.dev.yml` |
| Models | `./download_models.sh` (~16 GB total incl. the LLM, SHA-256 verified) |
| Auth | Disabled (`SZCA_API_KEY` unset) — never expose beyond localhost |
| TLS | None (localhost only) |

**Startup:**
```bash
./download_models.sh                       # ~16 GB, SHA-256 verified
cp env.dev.example .env.dev                # then edit ORT_DYLIB_PATH for your OS
set -a && . ./.env.dev && set +a
cargo run --manifest-path szca_media_gateway/Cargo.toml
curl http://localhost:3000/health
curl http://localhost:3000/v1/pools        # per-stage replicas + queue depth
```

**Verify the real dev LLM** (loads 14 GB; use `--release`, it is ~50× faster):
```bash
ORT_DYLIB_PATH=/opt/homebrew/lib/libonnxruntime.dylib \
LLM_MODEL_DIR=$PWD/models/llm/Hermes-3-Llama-3.2-3B \
cargo test --release --manifest-path szca_media_gateway/Cargo.toml \
  --test llm_real_inference -- --nocapture --test-threads=1
```
Asserts the checkpoint's own chat template + stop tokens are used, that deltas
stream and reconstruct the reply, that generation stops on EOS rather than the
token cap, and that barge-in cancels within one token. Skips cleanly (green) when
`LLM_MODEL_DIR` is unset, so CI without weights still passes.

### PROD Profile — vLLM GPU, Enterprise, 1,000+ Sessions

| Property | Value |
|----------|-------|
| Hardware | **g6e.48xlarge** — 8× L40S (48 GB each), 192 vCPUs, 1.5 TB RAM, 400 Gbps networking |
| LLM backend | `LLM_BACKEND=vllm` → vLLM on all 8 L40S via tensor parallelism |
| LLM model | Hermes-3-Llama-3.1-8B FP8 (~8 GB, ~256 MB KV/session) |
| STT | Parakeet TDT 0.6B int8, in-process ORT, 32 replicas (~10 GB) — OR on 1 L40S via Triton |
| TTS | Kokoro-82M + Misaki, in-process ORT, 16 replicas (~2.5 GB) — OR on 1 L40S |
| Concurrent sessions | **1,000+** target capacity on 192 vCPUs + 8× L40S GPUs (gpu-memory-utilization=0.85) |
| WebSocket | `wss://api.yourdomain.com/v1/realtime?dialect=openai\|gemini` |
| Pool sizing | `STT_REPLICAS=48 LLM_REPLICAS=128 TTS_REPLICAS=24` |
| Queue backlog | `SZCA_QUEUE_BACKLOG=1024` (prevents 503 waves on turn bursts) |
| Tokio threads | `SZCA_WORKER_THREADS=32 SZCA_BLOCKING_THREADS=256` |
| Deployment | Helm Chart (`deploy/charts/szca-media-gateway/`) on EKS |
| Load balancer | AWS ALB Ingress (sticky sessions, TLS termination) |
| Admission | Per-pod `SessionManager` in-process atomic counter (default `1000` per pod) |
| Auth | `SZCA_API_KEY` set → Bearer token required |
| TLS | AWS ALB / Nginx Ingress terminates (443/wss), workers plain HTTP |
| Observability | Prometheus `/metrics` scraping + Grafana (`deploy/grafana/`) |
| Load target | 1,000 concurrent sessions across pod replicas (target validation on GPU) |

**Deploy sequence:**
```bash
docker compose -f docker-compose.prod.yml up -d        # Workers take 3-5 min
docker compose -f docker-compose.prod.yml logs -f worker1  # Wait for "ready"
curl -k https://localhost/health
curl -k https://localhost/v1/pools
locust -f locustfile.py --host https://localhost --users 300 --spawn-rate 10 --headless
python3 baseline_check.py results/load_300_stats.csv
```

### Dev → Prod Parity

| Property | Dev | Prod | Same? |
|----------|-----|------|-------|
| STT model | Parakeet 0.6B int8 | Parakeet 0.6B int8 | ✅ |
| TTS model | Kokoro-82M + Misaki | Kokoro-82M + Misaki | ✅ |
| VAD model | Silero v5 ONNX | Silero v5 ONNX | ✅ |
| LLM model | Hermes-3-3B FP32 (ONNX) | Hermes-3-8B FP8 (vLLM) | ✅ Same family |
| LLM chat template | ChatML (detected) | ChatML (server-side) | ✅ |
| Default system prompt | model-agnostic | model-agnostic | ✅ (asserted by test) |
| Turn event contract | `run_turn` | `run_turn` | ✅ (one impl, e2e tested) |
| Pool architecture | `StagePool<R>` | `StagePool<R>` | ✅ |
| WebSocket API | OpenAI + Gemini dialect | OpenAI + Gemini dialect | ✅ |
| HTTP API | `/v1/stt|llm|tts` | `/v1/stt|llm|tts` | ✅ |
| LLM SSE format | `choices[0].delta.content` | `choices[0].delta.content` | ✅ (contract tested) |
| Barge-in / cancel | `AtomicBool` flag | `AtomicBool` flag | ✅ |
| Graceful shutdown | SIGTERM + 30s drain | SIGTERM + 30s drain | ✅ |
| Request timeout | 60s TimeoutLayer | 60s TimeoutLayer | ✅ |

---

## 5. Environment Variables

### Gateway

| `SZCA_QUEUE_BACKLOG` | 64 | Max MPMC queue capacity per stage pool before backpressure (503) |
| `SZCA_WORKER_THREADS` | num_cpus | Tokio worker thread pool size |
| `SZCA_BLOCKING_THREADS` | 512 | Tokio blocking thread pool size for pool workers |

### Production Runtime Tuning Guide

#### 1. Worker & Thread Pool Sizing
- **`SZCA_WORKER_THREADS`**: Controls Tokio's async worker threads. Set to physical CPU core count (e.g. `192` on `g6e.48xlarge`).
- **`SZCA_BLOCKING_THREADS`**: Slices `spawn_blocking` pool for model replicas. Must exceed `STT_REPLICAS + LLM_REPLICAS + TTS_REPLICAS + max_sessions`. Default `512` is recommended.

#### 2. Replica Sizing per GPU (e.g., L40S 48GB)
- **`STT_REPLICAS`**: Each Parakeet/Sherpa replica uses ~1.5 GB VRAM. At ~200ms per utterance chunk, 1 replica handles ~5 utterances/sec. For 300 active sessions with an average 5s turn interval (~60 utterances/sec total load), **12-16 STT replicas** are required.
- **`LLM_REPLICAS`**: Each INT8 ONNX LLM replica uses ~3 GB VRAM (Qwen2.5-1.5B) or routes via external vLLM worker pool. Target `4-8` replicas per GPU.
- **`TTS_REPLICAS`**: Each Kokoro TTS replica uses ~500 MB VRAM. At ~50ms synthesis time per audio chunk, **16-32 TTS replicas** handle up to 300+ concurrent streaming voices.

#### 3. Production STT Backend Choice
In production (`STT_BACKEND=streaming`), the gateway uses **Parakeet EOU 120M** (FP16 streaming FastConformer encoder) for turn-end detection via the `<EOU>` token, paired with **Sherpa-Zipformer** streaming STT for low-latency acoustic decoding.

#### 4. Queue Backlog & Admission Control
- **`SZCA_QUEUE_BACKLOG`**: Maximum queued jobs before returning `503 Service Unavailable`. Under high load, set `64-128`.
- **`SZCA_MAX_SESSIONS`**: Maximum concurrent WebSocket connections per gateway instance (default `1000`).
| `LLM_MAX_TOKENS` | 1024 | Max tokens per generation (**vLLM backend only**) |
| `LLM_TEMPERATURE` | 0.7 | Sampling temperature (**vLLM backend only**) |
| `LLM_MODEL_DIR` | `./models/llm` | Model directory (**onnx backend only**) |
| `LLM_ONNX_FILE` | first `*.onnx` in dir | Graph filename override |
| `LLM_TOKENIZER_FILE` | first `*tokenizer*.json` | Tokenizer filename override |
| `LLM_MAX_NEW_TOKENS` | 256 | Per-turn token cap (**onnx backend only**) |
| `LLM_CHAT_TEMPLATE` | auto-detect | Force `chatml` or `llama` (**onnx backend only**) |
| `STT_BACKEND` | `parakeet` | `parakeet` (full-utterance TDT 0.6B), `streaming`/`eou` (cache-aware EOU 120M), or `zipformer` (Sherpa Zipformer). Any other value falls back to `parakeet` |
| `STT_MODEL_DIR` | `./models/stt` | Parakeet model directory (**parakeet backend only**) |
| `STT_EOU_MODEL_DIR` | `./models/stt_eou` | Streaming EOU model directory (**streaming backend only**) |
| `SHERPA_MODEL_DIR` | `./models/sherpa_zipformer` | Sherpa Zipformer model directory (**zipformer backend only**) |
| `TTS_MODEL_DIR` | `./models/tts` | Kokoro model directory |
| `TTS_VOICE` | `af_heart` | Kokoro voice pack name |
| `SZCA_LISTEN_ADDR` | `0.0.0.0` | Bind address |
| `SZCA_PORT` | `3000` | Listen port |
| `SZCA_MAX_SESSIONS` | 1000 | Admission-control cap (503 past this) |
| `SZCA_API_KEY` | (disabled) | Bearer token for API auth |
| `RUST_LOG` | (none) | Log level: `info`, `debug`, `trace` |
| `SILERO_VAD_MODEL` | (empty → RMS fallback) | Path to Silero VAD ONNX model |
| `ORT_DYLIB_PATH` | auto-detect | Path to `libonnxruntime.so/dylib` |

`LLM_MODEL` is overloaded on purpose: it names the **served model** for the vLLM
backend, and selects the **download variant** (`hermes3` / `qwen25` /
`llama32-1b`) in `download_models.sh`. The gateway only reads it under
`LLM_BACKEND=vllm`, so there is no conflict at runtime.

Ready-to-copy templates: **`env.dev.example`** and **`env.prod.example`**
(`.env*` is gitignored; the `.example` files are committed).

---

## 6. API Reference

### WebSocket — Realtime Voice

```
ws://localhost:3000/v1/realtime?dialect=openai|gemini
```

Bidirectional streaming: audio in → STT → LLM → TTS → audio out. Supports OpenAI Realtime and Gemini Live wire dialects.

### HTTP — Standalone Services

```bash
# STT (streaming transcript)
curl -X POST http://localhost:3000/v1/stt/stream \
  -H "Content-Type: application/json" \
  -d '{"audio": "<base64-pcm16>", "language": "en"}'

# LLM (streaming tokens)
curl -X POST http://localhost:3000/v1/llm/stream \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "Hello"}], "stream": true}'

# TTS (streaming audio)
curl -X POST http://localhost:3000/v1/tts/stream \
  -H "Content-Type: application/json" \
  -d '{"input": "Hello world", "voice": "af_heart"}'

# Pool health
curl http://localhost:3000/v1/pools

# Prometheus metrics
curl http://localhost:3000/metrics

# Health check
curl http://localhost:3000/health
```

All three `/v1/*/stream` endpoints are **Server-Sent Events** and emit
incrementally as inference produces output, then one terminal event:

| Endpoint | Incremental | Terminal |
|----------|-------------|----------|
| `/v1/stt/stream` | `partial` — interim transcripts, suppressed by `"interim_results": false` | `final` — `{text, confidence, timestamp}` |
| `/v1/llm/stream` | `token` — `{text, token_id, logprob, index}` | `eos` — `{text, total_tokens, finish_reason}` |
| `/v1/tts/stream` | `audio_chunk` — `{pcm, sample_rate, duration_ms, sequence}`, `pcm` = base64 PCM16 mono | `eos` — `{total_chunks, total_duration_ms, sample_rate}` |

Any endpoint may instead emit one terminal `error` event (`{"error": "..."}`) on a
full queue, a closed pool, or a panicking inference task. SSE headers are already
flushed at that point, so the status is `200` regardless — clients must key off
the event name, not the status code.

`finish_reason` is `stop` (model emitted its own EOS) or `length` (cut off by the
request's `max_tokens`). Client disconnect sets the job's cancel flag, so the
replica stops generating rather than finishing output nobody reads. `token_id` /
`logprob` on `token` events are `-1` / `0.0` placeholders — the pool surfaces
decoded text, not per-token ids or probabilities.

---

## 7. Hardware Sizing

### Per-Stage Cost

| Stage | Per-Replica RAM | CPU per Replica | Inference Time |
|-------|----------------|-----------------|----------------|
| VAD (Silero) | ~30 MB | ~1 core bursty | ~5 ms/frame |
| STT (Parakeet) | ~300 MB | 2–4 cores | ~200 ms/utterance |
| LLM (Qwen 1.5B, CPU) | ~1.5 GB | 4–8 cores | ~10 tok/s |
| TTS (Kokoro) | ~150 MB | 1–2 cores | ~80 ms/sentence |

### KV Cache Math (vLLM, Llama 3.1 8B)

```
KV per token (FP8)  = 2 × 32 layers × 8 KV heads × 128 head_dim × 1 byte = 64 KB
KV per user @ 4096 tokens = 256 MB
```

| GPU | VRAM | Model (FP8) | Overhead | KV Budget | Max Sessions |
|-----|------|-------------|----------|-----------|--------------|
| L40S 48GB | 48 GB | 8 GB | 8 GB | 32 GB | ~125 |
| A100 80GB | 80 GB | 8 GB | 8 GB | 64 GB | ~250 |

### Scaling Table

| Hardware | Concurrent | Throughput | Monthly Cost |
|----------|-----------|------------|-------------|
| 1× L4 24GB | 100–200 | 2–3K tok/s | ~$200 |
| 1× A10G 24GB | 200–400 | 4–6K tok/s | ~$350 |
| **1× A100 80GB** | **500–800** | **10–15K tok/s** | **~$1,700** |
| 1× H100 80GB | 800–1,200 | 15–25K tok/s | ~$3,000 |

---

## 8. Cost Analysis

### Monthly (500–800 Concurrent, On-Prem)

| Item | SZCA (1× A100) | OpenAI Realtime |
|------|----------------|-----------------|
| GPU (amortized) | ~$1,500/mo | Included |
| Power + Cooling | ~$200/mo | N/A |
| **Total** | **~$1,700/mo** | **~$1,296,000/mo** |
| Cost/Minute | ~$0.00007 | ~$0.06 |
| **Savings** | **99.87%** | Baseline |

---

## 9. Dev Setup

See [DEVELOPMENT.md](DEVELOPMENT.md) for the full guide.

```bash
# 1. Models (~16 GB — the FP32 dev LLM is 14.4 GB of that). SHA-256 verified.
#    Everything lands under ONE root: models/{stt,tts,llm,vad,dfn3}, with one
#    directory per LLM under llm/. Those are the gateway's built-in defaults, so
#    running from the repo root needs no path env vars.
#    Swap the LLM variant with LLM_MODEL=qwen25|llama32-1b if you want a small
#    model for quick iteration (different family than prod — see §4).
./download_models.sh

# 2. Env (edit ORT_DYLIB_PATH for your OS; .env* is gitignored)
cp env.dev.example .env.dev
set -a && . ./.env.dev && set +a

# 3. Run
cargo run --manifest-path szca_media_gateway/Cargo.toml   # listens on :3000

# 4. Verify
curl http://localhost:3000/health
curl http://localhost:3000/v1/pools     # per-stage replicas + queue depth
```

The startup log states which backend and template were chosen — check it before
debugging output quality:

```
INFO Building LLM pool (ONNX)
INFO LLM chat template selected chat_template=ChatML
INFO ONNX causal-LM loaded n_layers=28 n_kv_heads=8 head_dim=128
     chat_template=ChatML eos_ids=[128001, 128008, 128009, 128039]
```

If a stage's model is missing, its pool build fails and sessions fall back to
**stub stages** (deterministic placeholder text/audio) rather than erroring — so
"it runs but says `[stub-llm] …`" means weights weren't found, not that the
pipeline is broken.

Docker:
```bash
./download_models.sh   # on the HOST first; the volume is shared into the container
docker compose -f docker-compose.dev.yml up --build
```

---

## 10. Prod Setup

### vLLM Worker Startup

```bash
vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --host 0.0.0.0 --port 8000 \
  --dtype float16 --quantization fp8 --kv-cache-dtype fp8 \
  --gpu-memory-utilization 0.88 \    # 0.85 for L40S
  --max-num-seqs 250 --max-model-len 4096 \
  --enable-chunked-prefill --chunked-prefill-max-tokens 512 \
  --enable-prefix-caching --tensor-parallel-size 1
```

**Note on `--dtype float16` + `--quantization fp8`:** `--dtype` sets compute precision for activations (FP16). `--quantization fp8` quantizes only **weights** to FP8. Activations remain FP16. Model footprint ~8 GB; compute is FP16.

**Model loading:** vLLM with FP8 on A100 takes **3–5 minutes** to load + compile CUDA graphs. Healthcheck `start_period: 360s`.

**GPU utilization:** A100 HBM2e safe at 0.88; L40S GDDR6X safe at 0.85.

### Nginx Config (key elements)

```nginx
upstream vllm_workers {
    hash $cookie_vllm_worker consistent;  # sticky for prefix cache
    least_conn;
    server gpu-worker-1:8000 max_fails=3 fail_timeout=30s;  # passive health
    server gpu-worker-2:8001 max_fails=3 fail_timeout=30s;
    keepalive 64;
}
```

### Kubernetes Deployment & TLS Termination

When deploying the `szca_media_gateway` to production via Kubernetes (k8s), **TLS termination should NOT be handled by the Rust application itself**.

- **Ingress Level:** Place the gateway behind an Ingress controller (e.g., NGINX Ingress). Configure the Ingress to terminate TLS (`https://` or `wss://`) and forward plain TCP/HTTP to the gateway pods.
- **Certificate Management:** Use a tool like `cert-manager` to automatically provision and rotate Let's Encrypt certificates at the Ingress level. The gateway pods do not need to mount or manage certificates.
- **Performance:** This offloads the heavy CPU overhead of TLS handshakes and decryption to the optimized Ingress controller, reserving the gateway's compute exclusively for real-time audio and model processing.

### Docker Compose

```bash
docker compose -f docker-compose.prod.yml build
docker compose -f docker-compose.prod.yml up -d
```

---

## 11. Testing

### Test Status

- **212 unit tests** — all passing
- **25 integration tests** — all passing:
  - `tests/e2e_pipeline.rs` (5) — full turn PCM→STT→LLM→TTS→PCM through the real
    `run_turn` with stub stages: event-order contract, LLM‖TTS interleaving,
    delta→final reconstruction, pre-turn + mid-generation barge-in, both wire
    dialects, VAD utterance boundary. **No weights or network needed.**
  - `tests/llm_real_inference.rs` (3) — real dev LLM: streams, stops on its own
    EOS, cancels on barge-in. Skips green without `LLM_MODEL_DIR`.
  - `tests/stt_eou_real_inference.rs` (7) — real streaming Parakeet EOU: decodes
    `"hello world"` directly, pooled, and 20 ms frame-by-frame; asserts agreement
    across paths, reset() repeatability, silence stays empty, RTF >2×.
    Skips green without `STT_EOU_MODEL_DIR`.
  - `tests/stt_zipformer_real_inference.rs` (6) — real Sherpa Zipformer: decodes
    `"Hello World"` directly, pooled, and 20 ms frame-by-frame; RTF, silence.
    Skips green without `SHERPA_MODEL_DIR`.
  - `tests/silero_real_inference.rs` (2), `tests/dfn3_real_inference.rs` (2) —
    skip green without weights.
- **1 doc-test** — passing
- **Build**: `cargo check --all-targets` clean, **zero warnings**, **zero errors**

```bash
cargo test --manifest-path szca_media_gateway/Cargo.toml   # 212 + 25 + 1, no weights needed
```

The e2e test drives the **same** `rt_session::run_turn` the WebSocket worker
calls, so the per-turn event contract has exactly one implementation and cannot
drift between production and test.

### Load Testing

```bash
# HTTP endpoints (k6)
k6 run szca_load_test/load_test.js

# WebSocket realtime (k6)
k6 run szca_load_test/ws_load_test.js

# Enterprise scale (Locust) — locustfile.py is NOT in the repo yet; see §15
locust -f locustfile.py --host https://localhost --users 300 --spawn-rate 10 --headless
```

Step-load profile: 50→100→200→300 with per-step thresholds:

| Metric | 50 users | 100 users | 200 users | 300 users |
|--------|----------|-----------|-----------|-----------|
| TTFT (p95) | < 200ms | < 300ms | < 500ms | < 800ms |
| TPOT (p95) | < 20ms | < 25ms | < 30ms | < 40ms |
| HTTP 200 rate | > 99.9% | > 99.9% | > 99% | > 99% |

### Contract Test (Dev ↔ Prod Parity)

```bash
python3 contract_test.py http://localhost:8080 http://localhost:8000
```

### Chaos Scenarios

| Scenario | Expected |
|----------|----------|
| Worker OOM | Nginx routes to surviving worker within 30s |
| Redis failure | Admission falls back to accept-all (fail-open) |
| Context length spike | Hard cap enforced, no OOM |
| Barge-in / connection drop | GPU releases KV slot within 1 heartbeat |

### Performance Regression Baseline

```bash
python3 baseline_check.py results/load_300_stats.csv
# 20% regression tolerance per metric
```

---

## 12. CI/CD

GitHub Actions — 10 jobs:

| Job | What |
|-----|------|
| `rust-tests` | `cargo test` + `cargo clippy -- -D warnings` + `cargo fmt --check` |
| `cpp-tests` | CMake build + `ctest` |
| `integration-tests` | Integration + E2E tests |
| `security-tests` | Security test suite |
| `performance-tests` | Benchmark suite |
| `cargo-audit` | Dependency vulnerability scan |
| `secret-scan` | gitleaks secret scanning |
| `docker-build` | CPU + GPU image build + smoke test |
| `trivy-scan` | Container vulnerability scan |
| `summary` | Gate — all jobs must pass |

---

## 13. Runtime Abstraction (Don't Lock Into vLLM)

Our application never knows which runtime is behind the LLM interface. This is already
implemented via the `Replica` trait + `LlmBackend` enum:

```
              LLM Interface (Replica trait)
                       │
         ┌─────────────┼──────────────┐
         │             │              │
       vLLM        ONNX Runtime    Future Engine
    (prod GPU)    (dev CPU)       (just add variant)
```

```rust
// In our code — application code never changes:
response = llm.generate(prompt, instructions, cancel, on_token);

// Whether that goes to vLLM or ONNX is decided by config, not code:
// LLM_BACKEND=vllm  → LlmBackend::Vllm(VllmClient)  → /v1/chat/completions
// LLM_BACKEND=onnx  → LlmBackend::Onnx(QwenLlm)     → ORT Session::run()
```

**When ONNX Runtime becomes competitive for LLM inference** (or if you build your own
inference engine): swap the backend with zero application changes. The `Replica` trait
is the seam. Add a new variant to `LlmBackend`, implement `Replica`, done.

**Models roadmap:**

| Environment | Model | Runtime | Status |
|-------------|-------|---------|--------|
| Dev / CI | Hermes-3-Llama-3.2-3B ONNX | ONNX Runtime | 🔄 Switch from Qwen |
| Local dev | Hermes-3-Llama-3.2-3B ONNX | ONNX Runtime | 🔄 Switch from Qwen |
| Benchmark | Hermes-3-Llama-3.1-8B | vLLM | ✅ |
| Production | Hermes-3-Llama-3.1-8B | vLLM | ✅ |
| Future | Hermes-3-Llama-3.1-8B ONNX | ONNX Runtime | Research |

---

## 14. Key Decisions (Archived)

- **espeak-ng FFI → `misaki-rs`**: Pure-Rust Misaki port. Fixes phoneme-alphabet mismatch, removes C dependency.
- **ort 2.0 borrow pattern**: `Session::run()` borrows `self` mutably. Extract owned types in scoped blocks, drop outputs, then call other `&mut self` methods.
- **Sentence-chunked TTS**: LLM and TTS run via disjoint field borrows, no cross-thread state, single `AtomicBool` cancel.
- **vLLM for LLM at scale**: We do NOT build our own continuous batching engine. Our engine owns STT/TTS/VAD; vLLM owns token generation.
- **Runtime abstraction**: `LLM_BACKEND` config switch = `onnx` (in-process, CPU, dev) vs `vllm` (external GPU, prod). Same API contract. The `Replica` trait + `LlmBackend` enum means adding a third runtime later is just a new variant — zero application changes.
- **Same model family, different sizes**: Dev uses Hermes-3-3B ONNX, prod uses Hermes-3-8B on vLLM (quality + concurrency). Same family = no behavioral surprises between environments. Note dev's FP32 export trades speed for that parity — see §2.
- **Chat template comes from the checkpoint, never from the architecture**: Hermes-3-Llama-3.2-3B reports `model_type: "llama"` but is ChatML-tuned, so `config.json` says nothing about the prompt format. We read the model's own Jinja `chat_template` from `tokenizer_config.json` and probe which special tokens it emits. A `model_type` heuristic would have silently sent Llama-3 header tokens to a ChatML model — degraded replies with no error.
- **Stop tokens are a union, not a priority order**: Hermes-3's `generation_config.eos_token_id` is inherited from the Llama base (`128001/128008/128009`) while `config.eos_token_id` is `128039` (`<|im_end|>`) — the token the fine-tune actually ends turns with. "generation_config wins" would drop the real stop token and every reply would run to the token cap. Extra ids are harmless; a missing one is not.
- **Model-agnostic default system prompt**: hard-coding a vendor persona (the old Qwen default) leaks into whatever model is loaded. One neutral default for both templates, asserted by test, so a dev/prod backend swap can't change the assistant's behavior.
- **One `run_turn` for production and tests**: the per-turn event contract (transcript/text/audio deltas, terminal done-vs-cancelled, sentence interleaving) has a single implementation, driven with pool adapters in production and stub stages in `tests/e2e_pipeline.rs`. A test that reimplemented the sequence would pass while production drifted.
- **Pool adapters implement stage traits**: `SttPoolAdapter` implements `SttStage`, so session code calls `stt.transcribe(...)` unchanged.
- **Data Parallelism over Tensor Parallelism**: An 8B model fits on a single GPU; no TP needed.

---

## 15. What's Left (Backlog)

**Done since this list was written** (see §11 for how each is verified):

| # | Item | Status |
|---|------|--------|
| 1 | Switch dev model to Hermes-3-Llama-3.2-3B ONNX (tracking) | ✅ Done at the time. Later pivoted: Hermes-3-3B FP32 is too large for daily dev; Qwen2.5-1.5B INT8 is now the default. `download_models.sh` default is `qwen25`. Hermes-3-3B available via `LLM_MODEL=hermes3` for model-family testing. |
| 2 | End-to-end integration test (PCM→STT→LLM→TTS→PCM) | ✅ Done. `tests/e2e_pipeline.rs` (5 tests, no weights) + `tests/llm_real_inference.rs` (3 tests, real weights) |
| 3 | True LLM‖TTS overlap (parallel generation + synthesis) | ✅ Done. LLM runs in a scoped thread pushing sentences through a crossbeam channel; TTS runs on the calling thread (within `spawn_blocking`). Cuts turn latency: TTS synthesizes one sentence while LLM already generates the next. Preserves full event ordering contract (`TextDelta` → sentence → `AudioDelta` interleaving). 196 tests passing. |
| 4 | DFN3 noise cancellation (wired into pipeline) | ✅ Done. Uses `deep_filter` v0.2.5 crate (`transforms` feature) for STFT/iSTFT, ERB filterbank, band normalization. New module `dfn3_dsp.rs` bridges `Dfn3Model` (ORT inference) with `DFState` (analysis/synthesis). Chain: `16 kHz PCM → resample 48 kHz → DFState::analysis → ERB band_compr + band_mean_norm → DF band_unit_norm → Dfn3Model::run_flat → apply_band_gain + deep filter (5-tap convolution) → DFState::synthesis → resample 16 kHz`. Wired into main.rs `handle_voice_session`: creates `DspProcessor` with model_dir → wraps `Dfn3Dsp` → calls `dsp.process(pcm)` before VAD in `rt_session.rs`. Processing is per-session (ONNX sessions are per-replica but DFState/ring buffers are per-session). 203 tests passing, 0 warnings. |
| 6 | Per-pool latency histograms | ✅ Done. `hdrhistogram` crate, per-worker timing in `worker_loop`, `latency_snapshot()` → p50/p90/p95/p99 on `/v1/pools`. |
| 10 | Add Hermes-3-3B ONNX to `download_models.sh` | ✅ Downloads graph + external data + tokenizer + all three configs, each SHA-256 pinned |
| 11 | SHA-256 pin the Hermes-3 downloads | ✅ Done. All 6 files pinned (`MODEL_SHA256_HERMES3_*`); verified by a fresh download through the real `download_and_verify`. Two side findings fixed: metadata was being fetched from the repo **root** (the PyTorch checkpoint, `torch_dtype: bfloat16` / `use_cache: false`) instead of `onnx/`, and `tokenizer_config.json` — the file chat-template detection reads — was never downloaded at all |
| 8 | C++ build verification (cmake + ctest on target) | ✅ Done. `szca_onnx_engine` builds cleanly with cmake (7/7 CTest passing) on macOS with ONNX Runtime + SoXR auto-detected. |
| 9 | K8s deployment manifests (EKS Helm chart for prod) | ✅ Done. Full Helm chart at `deploy/charts/szca-media-gateway/` with Deployment, HPA, PDB, ALB Ingress, Service, PVCs, ConfigMap, ExternalSecret, ServiceAccount. Dev/prod value files at `deploy/values-*.yaml`. Chart passes `helm lint`. |

### P3 Backlog (not blocking production)

| # | Item | Effort | Notes |
|---|------|--------|-------|
| 3 | True LLM‖TTS overlap (parallel generation + synthesis) | ✅ Done. See progress table above. |
| 4 | DFN3 noise cancellation — wire into pipeline with real-time STFT/iSTFT reconstruction | ✅ Done. See progress table above. |
| 5 | Streaming STT — Parakeet EOU 120M + Sherpa Zipformer | ✅ Done | Three backends: parakeet, streaming (EOU), zipformer. See §16. |
| 6 | Per-pool latency histograms (p50/p95/p99) | ✅ Done. See progress table above. |
| 7 | Spanish TTS (non-English G2P engine) | **Phase 2** | misaki-rs is English-only. Punted — not blocking core voice pipeline. |
| 8 | C++ build verification (cmake + ctest on target) | ✅ Done. Built and tested `szca_onnx_engine` (7/7 CTest passing, ORT + SoXR auto-detected). Stable build in `deploy/charts/szca-media-gateway/`. |
| 9 | K8s deployment manifests (EKS Helm chart for prod) | ✅ Done. Full Helm chart: `deploy/charts/szca-media-gateway/` with Deployment, HPA, PDB, Ingress (ALB), Service, PVCs, ConfigMap, ExternalSecret, ServiceAccount. Dev/prod values in `deploy/values-{dev,prod}.yaml`. Chart passes `helm lint`. |
| 12 | Quantize the dev LLM → pivot: Qwen2.5-1.5B INT8 as default | ✅ Done. See progress table above. |
| 13 | 1000-session scale benchmarking on GPU (AWS g6e.48xlarge) | **Production Deployment Phase** | Reserved for GPU environment deployment using `szca_load_test/ws_load_test.js` or `locustfile.py`. **Blocker:** Requires AWS credentials and provisioning of a `g6e.48xlarge` by the DevOps team. |
| 14 | Upgrade `ort` to 2.0.0 stable | **Upstream Tech Debt** | Currently pinned to `=2.0.0-rc.10`. **Blocker:** Waiting for `ort` maintainers to publish the stable 2.0.0 release to crates.io. |
| 15 | Base image digest pinning (`@sha256:...`) in Dockerfile | ✅ Done | Pinned immutable sha256 digests in `Dockerfile` and `Dockerfile.dev`. |

> **Milestone Complete:** All Phase 1–6 features are shipped. The two remaining items (GPU load test, `ort` stable upgrade) are blocked on external factors — infrastructure access and upstream crate publication respectively. Phase 2 features (Spanish TTS, Hermes-3-3B INT8) are non-blocking enhancements with documented resource constraints.
>
> **Problems faced trying to quantize Hermes-3-3B:**
> 1. **OOM on quantize_dynamic loading:** `onnxruntime.quantization.quantize_dynamic()` calls `onnx.load(load_external_data=True)` which loads the entire 13 GB model + 1.2 MB graph into RAM as a contiguous protobuf, then ORT needs ~50% overhead during the quantization pass. With ~16 GB available RAM on a MacBook, the process was OOM-killed (exit code 137). Per-weight streaming approaches failed because the Python protobuf API doesn't support modifying external-data weights in-place — the graph structure references FP32 data types and the tooling expects the full model in memory.
> 2. **PyTorch dependency for optimum-cli:** The recommended tool `optimum-cli` requires PyTorch to load the HuggingFace model and re-export with quantization. PyTorch is a ~2-3 GB install with native code compilation, which was blocked by Claude Code's auto-mode safety classifier for system-level pip installs with native extensions.
> 3. **Qwen2.5-1.5B already works as INT8:** The `models/llm/Qwen2.5-1.5B-Instruct/` directory already contained a self-contained 1.5 GB INT8 ONNX from a prior download — no external data file, no OOM risk, loads in ~4 seconds, ~10-20 tok/s on CPU, and compatible ChatML format. Promoting it was a 15-minute configuration change vs a 1-day toolchain fight.
>
> **Resolution:** Switched the default dev LLM from Hermes-3-3B FP32 to Qwen2.5-1.5B INT8. Hermes-3-3B remains available via `LLM_MODEL=hermes3` for model-family parity testing. Any future INT8 export of Hermes-3-3B needs a machine with 32+ GB RAM to run `pip install optimum[onnxruntime] && optimum-cli export onnx --model NousResearch/Hermes-3-Llama-3.2-3B --quantize int8 ./output`. |

---

## 16. Dependency Audit — Streaming STT (Parakeet EOU 120M)

Streaming STT is a **core requirement**, not an optimization. This section records what was
measured before any code was written, because the first two answers we reached were wrong.

### The constraint

| Option | What changes | Blast radius |
|--------|-------------|--------------|
| **A: Bump ort rc.10 → rc.12** + add `parakeet-rs` crate | `Cargo.toml`, both Dockerfiles (tarball SHAs), every ORT call site | **HIGH** |
| **B: Slim wrapper** using our existing ort rc.10 | One new file `src/rt_stt_eou.rs` | **NONE** |

### Audit findings — why Option A is rejected

| Finding | Detail |
|---------|--------|
| 🚨 **`ort` rc.10→rc.12 is BREAKING** | Confirmed by source inspection: the `inputs!` macro is removed entirely. rc.12 restructured the API (`value.rs` → `value/`, `session.rs` → `session/`, only `ortsys!` remains). We use `ort::inputs!` across **6 files** (rt_stt, rt_llm, rt_tts, dfn3, silero). Every site would need rewriting. |
| 🚨 **ndarray 0.16 vs 0.17** | We use 0.16; ort rc.12 and parakeet-rs both use 0.17. Bumping ort forces ndarray 0.17 across our codebase. |
| ⚠️ **ort-sys edition 2024** | ort rc.12 + parakeet-rs use `edition = "2024"` (rustc ≥1.85; we have 1.92 — fine today). |
| ⚠️ **Dockerfile.dev ORT tarball** | Pinned to ORT 1.22.0 with SHA-256; a bump needs new SHAs verifiable only on the target platform. |
| ⚠️ **Dockerfile prod uses apt** | Prod installs `libonnxruntime-dev` from apt — a different version path than the tarball. |
| 🚨 **`parakeet-rs` serializes inference** | `ParakeetEOUHandle` holds `Arc<Mutex<ParakeetEOUModel>>` and holds the lock across the whole decode loop. One session behind a mutex contradicts `StagePool`'s one-model-per-thread design; at 1k sessions it is a hard serialization point. |

### 🚨 The finding that changed the plan: ORT 1.22 cannot run the INT8 encoder

The `soniqo` INT8 encoder uses **`ConvInteger`** (56 nodes) and `MatMulInteger` (154). ORT has
no CPU kernel for `ConvInteger` with signed INT8 until **1.24**. Measured directly:

| ONNX Runtime | Loads `soniqo` INT8 encoder? |
|--------------|------------------------------|
| 1.19.2 | ❌ `NOT_IMPLEMENTED: ConvInteger(10)` |
| **1.22.0 / 1.22.1** ← **our pin** (`ort` rc.10 → API 22) | ❌ `NOT_IMPLEMENTED: ConvInteger(10)` |
| 1.23.0 | ❌ `NOT_IMPLEMENTED: ConvInteger(10)` |
| 1.24.1 / 1.24.4 / 1.25 / 1.26 / 1.27 / 1.28 | ✅ loads and decodes |

**Verified on both `arm64` (macOS host) and `x86_64` (Linux container).** It is not a Mac
quirk — it would fail identically on the g6e.48xlarge. This invalidates *both* original
options: Option B's premise was "wrapper on existing ort", but the chosen **model** needs
ORT ≥1.24 regardless of how we call it. A slim wrapper against this checkpoint would have
compiled, loaded, and then failed at session creation in production.

### The resolution: a different export, not a different runtime

`AIsley/parakeet-realtime-eou-120m-streaming-fp16` publishes the same NVIDIA base model
(`nvidia/parakeet_realtime_eou_120m-v1`) as an **FP16 encoder**, which avoids `ConvInteger`
entirely. Verified end-to-end on **ORT 1.22.0** — our exact pinned version:

```
ort 1.22.0, FP16 encoder + int8 decoder_joint
  TEXT: 'hello world'   eou_at=(chunk 0, frame 16)
```

| Property | `soniqo` INT8 (rejected) | `AIsley` FP16 (chosen) |
|----------|--------------------------|------------------------|
| Min ORT | **1.24** ❌ | **1.22** ✅ (our pin) |
| Graphs | 3 (encoder + decoder + joint) | **2** (encoder + fused `decoder_joint`) |
| Encoder size | 132 MB | 232 MB |
| Chunk | 64 mel frames (640 ms) | 128 mel frames (1.28 s) |
| `pre_cache` | `[1,128,9]` | `[1,128,16]` |
| cache dtype | `int64` lengths | `int32` lengths |
| Encoder output | `[1,T',512]` | `[1,512,T']` |
| Encoder speed (M-series CPU) | 23 ms/chunk (27× RTF) | 60 ms/chunk (**21× RTF**) |
| Decoder signature | separate 640-dim joint | **identical to our Parakeet TDT decoder** |

The FP16 encoder logs `Could not find a CPU kernel ... constant fold MatMul` warnings on
1.22 — cosmetic; it still runs at 21× realtime, comfortably inside the 1.28 s chunk window.
Its fused `decoder_joint` takes `encoder_outputs`/`targets`/`input_states_{1,2}` — the *same
signature* `rt_stt.rs` already drives, so the decode loop is a known pattern, not new work.

### Verified feature contract (this is the part that silently breaks)

A Python reference decode (`onnxruntime` + numpy) was built first to pin the mel contract by
experiment. Normalization was the trap:

| Mel normalization | Decoded text |
|-------------------|--------------|
| `per_feature` (NeMo's default, per-utterance mean/var) | `''` — **0 tokens** |
| global mean/var | `''` — **0 tokens** |
| **none — raw `log(mel + 2⁻²⁴)`** | ✅ `'hello world'` |

This is the opposite of what our full-utterance Parakeet TDT frontend does: the
`nemo128.onnx` graph contains a `normalize` local function applying per-utterance mean/var.
**Reusing `nemo128.onnx` for streaming would normalize each chunk by its own statistics** and
produce empty or garbage transcripts with no error — so the streaming path needs its own
frontend. Confirmed parameters: 128 mels, nFFT 512, hop 160, win 400 (centered in the FFT
frame), periodic Hann, reflect-pad `nFFT/2`, Slaney-normalized triangular filterbank on the
HTK mel scale, `log(x + 2⁻²⁴)`, **no normalization**. Pre-emphasis 0.97 is in `config.json`
but made no difference to output; the log-guard does (`1e-5` truncated `'hello world'` →
`'hello'`).

Also verified: `<EOU>` (id 1024) fires ~1 s into trailing silence; transcript is stable at
30 dB and 20 dB SNR and degrades at 10 dB (which is what DFN3 is for).

### Decoder output slot — the second silent trap

`decoder_joint` returns `[1, 1, target_plus_sos, 1027]`. Slot `0` is the SOS position; the
**last** slot is the prediction for the supplied target. Reading slot 0 decodes
`'he wor worww'` instead of `'hello world'` — plausible-looking output, not an error.

### Recommendation: Option B (wrapper) on the FP16 export

Write `src/rt_stt_eou.rs` implementing `SttStage` against our existing **ort rc.10 / ORT
1.22**, plus a streaming mel frontend in Rust matching the contract above. No change to
`ort`, `ndarray`, either Dockerfile, or the wire protocol.

The ort rc.12 migration remains a real debt (we are exact-pinned to a release candidate) but
it is now decoupled from streaming STT: schedule it when ort 2.0 goes stable, as its own PR
with the test suite as the safety net — not as collateral damage from a feature.

### ✅ Shipped — what was actually built

**Parakeet EOU 120M**

| Item | Detail |
|------|--------|
| `src/rt_stt_mel.rs` | Streaming log-mel frontend. 128 mel / 512 nFFT / 160 hop / 400 win, periodic Hann centered in the FFT frame, reflect pad, Slaney-normalized HTK filterbank, `ln(x + 2⁻²⁴)`, **no normalization**. Zero per-frame allocation. |
| `src/rt_stt_eou.rs` | `ParakeetEouStt`: 2 ORT sessions, the 4 encoder caches carried across 1.28 s chunks, RNN-T greedy decode reading the LAST logit slot, `<EOU>`/`<EOB>`/blank handling. |
| Measured | 1.60 s audio in 0.105–0.154 s → **10–15× realtime**. `"hello world"` |

**Sherpa Zipformer**

| Item | Detail |
|------|--------|
| `src/rt_stt_zipformer.rs` | `SherpaZipformer`: 3 ORT sessions, 19-layer cache-aware encoder with 116 cache tensors, stateless LSTM decoder, separate joiner, Kaldi-style 80-mel frontend (Povey window, DC offset removal, peak=1 triangles). |
| Frontend | 80 mel, Povey window `pow(hann, 0.85)`, DC offset removal per frame, pre-emphasis 0.97, Kaldi peak=1 triangular filterbank (no area normalization). |
| Decode | Stateless: 2-token context window `[sos, blank]`, 650-token BPE vocab, `"token id"` per-line format. |
| Measured | 1.60 s audio in 0.078 s → **20.4× realtime**. `"Hello World"` |

**Shared wiring**

| Item | Detail |
|------|--------|
| `src/stage_pools.rs` | `SttBackend` enum (`Batch` \| `Streaming` \| `Zipformer`). `SttPool = StagePool<SttBackend>`, so the pool, the adapter, and all session code are unchanged across backends. |
| Selection | `STT_BACKEND=streaming` or `eou` for EOU; `zipformer` for Sherpa; anything else (including typos) falls back to Parakeet TDT. `GET /health` reports the resolved backend as `stt_backend`. |
| Dep delta | `realfft = "3.5"` promoted to direct dep (already in `Cargo.lock`). **No new crates, no version changes.** |

**Tests — why text assertions matter**

Every failure mode found while building these stages was SILENT — the model loads,
inference runs, no error is raised, and the transcript is simply empty or plausible
garbage:

| Mistake | Symptom | Which model |
|---|---|---|
| Any mel normalization | `""` — zero tokens | EOU |
| Log guard `1e-5` instead of `2⁻²⁴` | `"hello"` — truncated | EOU |
| Joint logit slot 0 instead of last | `"he wor worww"` — plausible garbage | EOU |
| Zero-dimension cache tensors | encoder fails silently | Zipformer |
| Pair-format vocab parser on line-index vocab | `""` — empty vocab | EOU |
| Caches not carried between chunks | chunk decoded as fresh utterance | Both |
| `reset()` missing between pool jobs | turn N+1 inherits N's context | Both |

### Completed Follow-Up Tasks

- **`<EOU>` early turn-end trigger live in `rt_session.rs`:** `rt_session` streams audio chunks incrementally into `SttStage` during inbound audio accumulation, emitting live `TranscriptDelta` events over WebSocket. When `<EOU>` is detected (`end_of_utterance == true`), the session triggers `maybe_start_response(...)` immediately (bypassing the ~500ms VAD silence timeout delay).
- **Production `Dockerfile` ORT Pinning:** Production `Dockerfile` now downloads, verifies (SHA-256), and pins ONNX Runtime 1.22.0 matching `Dockerfile.dev`.
- **Gated Streaming Model Downloads:** `download_models.sh` now gates EOU (~236 MB) and Zipformer (~156 MB) behind `--with-streaming` / `WITH_STREAMING=1`.

### WER Evaluation — LibriSpeech test-clean (174 utterances)

| Model | WER | Avg decode time | RTF | Notes |
|------|-----|----------------|-----|-------|
| **Parakeet TDT 0.6B** (default) | **3.28%** | 0.93 s | — | Full-context, highest accuracy |
| Sherpa Zipformer | 19.01% | 0.41 s | ~22× | English-only, 156 MB |
| Parakeet EOU 120M | 37.69% | 0.48 s | ~15× | Multilingual, 236 MB |

**Methodology:** 174 utterances from 4 speakers (6930, 1089, 61, 1188) of the
LibriSpeech test-clean corpus. WER computed via `jiwer` with case-insensitive,
punctuation-stripped comparison. Models run through Python ONNX Runtime wrappers
that replicate the Rust inference pipelines verbatim.

**Recommendation:** The default stays **`parakeet`** (TDT 0.6B). Its 3.28% WER is within the
expected range for a Conformer 0.6B on test-clean and comfortably beats the
streaming alternatives. `STT_BACKEND=zipformer` (19.01%) is a viable option when
peak latency or model size matters; `STT_BACKEND=eou` (37.69%) should only be used
when the pipeline requires the `<EOU>` signal and accuracy is secondary.

---

## 17. Source Map

| File | Purpose |
|------|---------|
| `src/stage_pool.rs` | Generic `StagePool<R>` core — queue, workers, cancel, backpressure |
| `src/stage_pools.rs` | Pool type aliases, adapters, `StagePools::from_env()`, `SttBackend` + `LlmBackend` enums |
| `src/rt_stt.rs` | Parakeet STT — 3-graph pipeline, TDT decode, Replica impl |
| `src/rt_stt_eou.rs` | Streaming STT — cache-aware Parakeet EOU 120M, RNN-T greedy decode, `<EOU>` detect, Replica impl |
| `src/rt_stt_zipformer.rs` | Streaming STT — Sherpa Zipformer 19-layer, Kaldi 80-mel frontend, stateless decoder, separate joiner, 116 cache tensors, Replica impl |
| `src/rt_stt_mel.rs` | Streaming log-mel frontend (128 mel, raw `ln(x + 2⁻²⁴)`, **no normalization**) |
| `src/rt_llm.rs` | In-process ONNX LLM — KV-cache decode, checkpoint-detected chat template + stop tokens, Replica impl |
| `src/rt_llm_client.rs` | vLLM streaming HTTP client, SSE parser, Replica impl |
| `src/rt_tts.rs` | Kokoro TTS — Misaki G2P, voice packs, sentence-chunked streaming, Replica impl |
| `src/rt_pipeline.rs` | `SttStage` / `LlmStage` / `TtsStage` traits + stubs |
| `src/rt_session.rs` | WS session loop — VAD → pools → barge-in → sentence interleaving; `run_turn` is the single per-turn event-contract impl |
| `src/rt_events.rs` | `ClientCommand` / `ServerEvent` neutral event model |
| `src/rt_protocol.rs` | OpenAI + Gemini dialect adapters |
| `src/api_routes.rs` | HTTP `/v1/stt|llm|tts/stream` + `/v1/pools` + `/metrics` |
| `src/main.rs` | Startup, pool init, graceful shutdown, signal handling |
| `src/metrics.rs` | Prometheus exporter |
| `src/vad.rs` | Silero VAD v5 + RMS fallback |
| `src/silero.rs` | Silero ONNX inference |
| `src/dfn3.rs` | DeepFilterNet3 noise cancellation |
| `src/dsp.rs` | Audio DSP utilities |
| `src/onnx.rs` | ORT initialization (load-dynamic) |
| `src/session.rs` | Session state machine + `SessionManager` |
| `src/ipc.rs` | POSIX shared-memory IPC |
| `src/protocol.rs` | Binary WS protocol (legacy) |
| `src/gateway.rs` | `GatewayConfig::from_env()` (`SZCA_*`) + legacy binary-WS frame handling |
| `src/ring_buffer.rs` | SPSC ring buffer |
| `tests/e2e_pipeline.rs` | E2E turn contract: event order, interleaving, barge-in, dialects, VAD (no weights) |
| `tests/llm_real_inference.rs` | Real dev-LLM inference: streaming, EOS stop, cancel (skips without weights) |
| `tests/stt_eou_real_inference.rs` | Real streaming EOU STT: decodes `hello world`, 7 tests (skips without weights) |
| `tests/stt_zipformer_real_inference.rs` | Real Sherpa Zipformer: decodes `Hello World`, 6 tests (skips without weights) |
| `tests/silero_real_inference.rs` | Real Silero VAD inference (skips without weights) |
| `tests/dfn3_real_inference.rs` | Real DFN3 3-stage chain (skips without weights) |
| `download_models.sh` | Hardened model download: pinned refs, SHA-256 verify, fails loudly |
| `env.dev.example` | DEV env template (ONNX/CPU, single replica) |
| `env.prod.example` | PROD env template (vLLM/GPU, replica pools, auth) |
| `DEVELOPMENT.md` | Developer quickstart |
| `PROJECT.md` | **This file — single source of truth** |
| `ARCHIVE_SCALING_PLAN.md` | Phase-by-phase evolution + decision reasoning (archived) |
| `docker-compose.dev.yml` | Dev stack |
| `Dockerfile.dev` | Dev gateway image |
| `Dockerfile.llm-dev` | Dev LLM server image |
| `dev_server.py` | FastAPI dev LLM server (optional; ONNX Runtime GenAI on CPU) |
| `szca_load_test/load_test.js` | k6 HTTP endpoint load test |
| `szca_load_test/ws_load_test.js` | k6 WebSocket realtime load test |

> Items referenced elsewhere in this doc that are **not yet in the repo**:
> `docker-compose.prod.yml`, `locustfile.py`, `contract_test.py`,
> `baseline_check.py`. The §10 / §11 command blocks that use them describe the
> intended prod workflow — treat them as a spec to implement, not as working
> commands.

---

## 18. Glossary

| Term | Meaning |
|------|---------|
| **Replica** | One loaded model instance on one OS thread, pulling jobs from a pool queue |
| **StagePool** | A bounded job queue + N replicas for one inference stage |
| **StagePools** | The shared set of all three pools (STT + LLM + TTS) |
| **PoolAdapter** | Implements stage traits by forwarding to a shared pool |
| **LlmBackend** | Enum: `Onnx(QwenLlm)` or `Vllm(VllmClient)` — selected by `LLM_BACKEND` |
| **ChatML** | `<|im_start|>role\n…<|im_end|>` prompt format (Qwen2/3, Hermes-3) |
| **External data** | ONNX weights stored in a sidecar file (`model.onnx_data`) resolved by a relative path recorded at export time |
| **Barge-in** | User interrupts AI speech; `AtomicBool` cancel stops all in-flight stages |
| **Sentence-chunked** | TTS synthesizes each sentence as LLM emits it, not after full reply |
| **PagedAttention** | vLLM's KV-cache management — pages allocated on demand, shared across sequences |
| **Prefix caching** | Reuse KV for identical system prompts across sessions (vLLM `--enable-prefix-caching`) |
| **TDT** | Token-and-Duration Transducer — Parakeet's decode algorithm with per-token duration |
| **FP8** | 8-bit floating point — halves model weight size vs FP16 |
| **KV-cache** | Key-Value tensors from previous tokens, reused to avoid recomputation |
