# SZCA Realtime Voice Engine — SRAM-Mesh Zero-Copy Architecture

Version 5.0.0 — A production Rust realtime voice engine: a bidirectional-streaming
alternative to OpenAI Realtime / Gemini Live, cascading **STT → LLM → TTS** with
server-side VAD, barge-in, and both wire dialects. Powered by a concurrent StagePool
architecture with config-driven LLM backend (in-process ONNX or external vLLM/TGI).

---

## gstack

Use the `/browse` skill from gstack for all web browsing — never use `mcp__claude-in-chrome__*`
tools directly. Available gstack skills (invoke via `/skill`):

| Category | Skills |
|----------|--------|
| **Product** | `/office-hours`, `/spec` |
| **Planning** | `/plan-ceo-review`, `/plan-eng-review`, `/plan-design-review`, `/plan-devex-review`, `/plan-tune` |
| **Design** | `/design-consultation`, `/design-shotgun`, `/design-html`, `/design-review` |
| **Quality** | `/review`, `/qa`, `/qa-only` |
| **Security** | `/cso` |
| **Debugging** | `/investigate` |
| **Shipping** | `/ship`, `/land-and-deploy`, `/canary`, `/benchmark` |
| **Documentation** | `/document-release`, `/document-generate` |
| **Retrospectives** | `/retro` |
| **DevEx** | `/devex-review` |
| **Browser/QA** | `/browse`, `/connect-chrome`, `/setup-browser-cookies` |
| **Setup** | `/setup-deploy`, `/setup-gbrain`, `/gstack-upgrade`, `/learn` |
| **Other** | `/autoplan`, `/codex`, `/careful`, `/freeze`, `/guard`, `/unfreeze`, `/context-restore`, `/context-save`, `/diagram`, `/scrape`, `/health` |

---

## Project Overview

**What it is:** A self-hosted, open-source voice inference platform that processes audio
in a **pure streaming pipeline** — no batching, no buffering, no waiting. Every audio
chunk flows through DSP → STT → LLM → TTS and returns to the client.

**Locked architecture:** Our Rust engine owns STT/TTS/VAD/orchestration; vLLM (or TGI)
owns LLM token generation on GPUs. We do NOT build our own LLM batching engine.

**Status:** Phases 1-6 Complete · Phase 7 Backlog (see `§15` in PROJECT.md)

---

## Architecture

### Design Principles
- **Zero-Copy IPC:** POSIX Shared Memory (`/dev/shm`) — no TCP loopback, no serialization
- **Lock-Free Hot Path:** Pre-allocated SPSC ring buffers, `AtomicBool` cancellation, 0-byte alloc on audio path
- **Pure Streaming:** Each token processed immediately — no accumulation, no batch wait
- **Binary Wire Protocol:** Raw PCM WebSocket frames — zero Base64, zero JSON for audio
- **Hardware-Agnostic:** ONNX Runtime EP abstraction — NVIDIA, AMD, Intel, Apple, CPU

### Pipeline
```
Mic 16kHz PCM → VAD (Silero) → STT (Parakeet TDT) → LLM (Hermes/vLLM) → TTS (Kokoro) → Speaker
```

### Model Stack
| Stage | Model | Precision | Size | License |
|-------|-------|-----------|------|---------|
| Noise suppression | DeepFilterNet3 | FP32 ONNX | ~10 MB | Apache-2.0 |
| VAD | Silero VAD v5 (16kHz) | FP32 ONNX | ~2 MB | MIT |
| STT (streaming) | Parakeet EOU 120M (cache-aware) | FP16 ONNX | ~236 MB | CC-BY-4.0 |
| STT (streaming) | Sherpa Zipformer 19-layer (cache-aware) | FP32 ONNX | ~156 MB | Apache-2.0 |
| LLM (dev) | Qwen2.5-1.5B-Instruct (default) | INT8 ONNX | ~1.5 GB | Apache 2.0 |
| LLM (dev alt) | Hermes-3-Llama-3.2-3B Instruct | FP32 ONNX | ~14.4 GB | Llama 3.2 Community |
| LLM (prod) | Hermes-3-Llama-3.1-8B Instruct | FP8 (vLLM) | ~8 GB | Llama 3.1 Community |
| TTS | Kokoro-82M v1.0 | FP16 ONNX | ~170 MB | Apache-2.0 |
| TTS G2P | Misaki (pure-Rust port) | — | — | MIT |

All models commercially usable (MIT / Apache 2.0 / CC-BY-4.0).

### 4-API Architecture
| API | Endpoint | Input | Output | Use Case |
|-----|----------|-------|--------|----------|
| **Unified Voice** | `wss://.../v1/realtime` | 16kHz PCM | 16kHz PCM | Real-time voice agent |
| **STT** | `POST /v1/stt/stream` | 16kHz PCM base64 | Partial + Final text | Transcription service |
| **LLM** | `POST /v1/llm/stream` | Text tokens | streamed `token`/`eos` events (SSE) | Chat/completion |
| **TTS** | `POST /v1/tts/stream` | Text tokens | streamed `audio_chunk`/`eos` events (SSE) | Speech synthesis |

### StagePool Architecture
- Generic `StagePool<R: Replica>` — bounded MPMC job queue (crossbeam-channel) + N replica threads
- Each replica = one model instance on one OS thread, shared by every session
- Per-stage replicas: `STT_REPLICAS`, `LLM_REPLICAS`, `TTS_REPLICAS` (0 = disabled)
- Pool adapters implement stage traits (`SttStage`, `LlmStage`, `TtsStage`) — session code never knows it's talking to a pool

### LLM Backend Abstraction
```rust
LLM_BACKEND=onnx  → LlmBackend::Onnx(QwenLlm)     → ORT Session::run() — dev CPU
LLM_BACKEND=vllm  → LlmBackend::Vllm(VllmClient)   → /v1/chat/completions — prod GPU
```
`LlmBackend` enum wraps either variant, selected by config, zero application code changes.

### Streaming Per-Stage Contract
| Stage | Input | Output | Interim | Final |
|-------|-------|--------|---------|-------|
| STT | batch utterance (post-VAD)¹ | streaming | `TranscriptDelta` (real word-boundary interims) | `TranscriptDone` |
| LLM | text | streaming | `TextDelta` (true token stream with UTF-8 holdback) | `TextDone` |
| TTS | text (sentence-chunked, parallel) | streaming | `AudioDelta` per sentence | `AudioDone` |

> **LLM‖TTS overlap**: LLM and TTS run in parallel — a scoped thread generates
> tokens and pushes sentences through a crossbeam channel while the calling thread
> synthesizes them concurrently. TTS starts before LLM finishes, cutting turn
> latency. See `run_turn()` in `rt_session.rs`.

¹ With the default `STT_BACKEND=parakeet`, STT input is the whole utterance — the
Conformer encoder is full-context (full-sequence attention + 8× subsampling).

`STT_BACKEND=streaming` selects the cache-aware **Parakeet EOU 120M** export instead
(`src/rt_stt_eou.rs`), which takes audio incrementally in 1.28 s chunks and carries its
attention/conv cache between calls — cost is linear in stream length rather than quadratic
in utterance length, and the model emits its own `<EOU>` end-of-utterance token. Measured
~15× realtime on CPU (1.6 s of audio in 105 ms). Wired through `SttBackend::Streaming` →
`StagePool` → `SttPoolAdapter`. `rt_session` feeds audio incrementally into `push_chunk()` and uses `<EOU>` as the early turn-end trigger, bypassing the ~500ms VAD silence timeout delay for sub-100ms turn-taking latency. See PROJECT.md §16.

---

## Deployment Profiles

The engine runs in two distinct profiles. Same codebase, same API contract — different
infra and capacity targets. The `LLM_BACKEND` env var switches between them.

### DEV Profile — ONNX-Only, CPU, Single-User

| Property | Value |
|----------|-------|
| Hardware | Any CPU (MacBook, Linux VM, CI runner) |
| LLM backend | `LLM_BACKEND=onnx` → in-process Qwen2.5-1.5B-Instruct via ORT |
| LLM model | Qwen2.5-1.5B-Instruct **INT8** ONNX (~1.5 GB self-contained). Also supports Hermes-3-Llama-3.2-3B FP32 via `LLM_MODEL=hermes3` (~14.4 GB) |
| LLM RAM | ~1.5 GB for Qwen2.5; ~15 GB for Hermes-3. Keep `LLM_REPLICAS=1` |
| Chat template | ChatML, auto-detected from `tokenizer_config.json` |
| STT | Parakeet TDT 0.6B int8, in-process ORT |
| TTS | Kokoro-82M + Misaki, in-process ORT |
| Concurrent sessions | 1–2 (single-user correctness) |
| WebSocket | `ws://localhost:3000/v1/realtime?dialect=openai\|gemini` |
| HTTP endpoints | `/v1/stt/stream`, `/v1/llm/stream`, `/v1/tts/stream` |
| Pool sizing | `STT_REPLICAS=1 LLM_REPLICAS=1 TTS_REPLICAS=1` |
| Docker | `docker-compose.dev.yml` (cargo-watch hot-reload) |
| Models | `./download_models.sh` (default: Qwen2.5, ~1.8 GB). `LLM_MODEL=hermes3` (~16 GB total) for prod-family testing |
| Auth | Disabled (`SZCA_API_KEY` unset) — never expose beyond localhost |
| TLS | None (localhost only) |

**Dev startup:**
```bash
./download_models.sh                       # ~16 GB, SHA-256 verified
cp env.dev.example .env.dev                # then edit ORT_DYLIB_PATH for your OS
set -a && . ./.env.dev && set +a
cargo run --manifest-path szca_media_gateway/Cargo.toml
curl http://localhost:3000/health
curl http://localhost:3000/v1/pools        # per-stage replicas + queue depth
```

**Verify real dev LLM** (loads 14 GB; use `--release`, ~50× faster):
```bash
ORT_DYLIB_PATH=/opt/homebrew/lib/libonnxruntime.dylib \
LLM_MODEL_DIR=$PWD/models/llm/Hermes-3-Llama-3.2-3B \
cargo test --release --manifest-path szca_media_gateway/Cargo.toml \
  --test llm_real_inference -- --nocapture --test-threads=1
```

### PROD Profile — vLLM, GPU, 1,000+ Sessions

| Property | Value |
|----------|-------|
| Hardware | **g6e.48xlarge** — 8× L40S (48 GB), 192 vCPUs, 1.5 TB RAM |
| LLM backend | `LLM_BACKEND=vllm` → external vLLM via `/v1/chat/completions` |
| LLM model | Hermes-3-Llama-3.1-8B FP8 (~8 GB, ~256 MB KV/session) |
| STT | Parakeet TDT 0.6B int8, in-process ORT (pool of 48 replicas) |
| TTS | Kokoro-82M + Misaki, in-process ORT (pool of 24 replicas) |
| Concurrent sessions | **1,000** |
| WebSocket | `wss://api.yourdomain.com/v1/realtime?dialect=openai\|gemini` |
| Pool sizing | `STT_REPLICAS=48 LLM_REPLICAS=128 TTS_REPLICAS=24` |
| Queue backlog | `SZCA_QUEUE_BACKLOG=1024` |
| Tokio threads | `SZCA_WORKER_THREADS=32 SZCA_BLOCKING_THREADS=256` |
| Docker | `docker-compose.prod.yml` → nginx + workers + redis |
| Load balancer | Nginx (sticky routing, passive health, 64 keepalive) |
| Admission | Redis-backed counter (fail-open on error) |
| Auth | `SZCA_API_KEY` set → Bearer token required |
| TLS | Nginx terminates (443), workers plain HTTP (trusted network) |
| Observability | Prometheus + DCGM GPU exporter + Nginx logs |
| Load tested | Locust step-load: 50→100→200→300 |

**Deploy sequence:**
```bash
docker compose -f docker-compose.prod.yml up -d        # vLLM workers take 3-5 min
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
| HTTP API | `/v1/stt\|llm\|tts` | `/v1/stt\|llm\|tts` | ✅ |
| LLM SSE format | `choices[0].delta.content` | `choices[0].delta.content` | ✅ (contract tested) |
| Barge-in / cancel | `AtomicBool` flag | `AtomicBool` flag | ✅ |
| Graceful shutdown | SIGTERM + 30s drain | SIGTERM + 30s drain | ✅ |
| Request timeout | 60s TimeoutLayer | 60s TimeoutLayer | ✅ |

Same binary, same wire contract — only `LLM_BACKEND`, pool sizes, and infra details change.

### Latency Budget — DEV (Hermes-3-3B ONNX, CPU)

| Stage | Technology | Latency |
|-------|-----------|---------|
| Noise Filtering | DeepFilterNet3 SIMD | ~1.5 ms |
| Speech Detection | Silero VAD v5 ONNX | ~0.5 ms |
| STT | Parakeet TDT 0.6B V3 INT8 | ~22 ms |
| Shared Memory IPC | POSIX Lock-Free SHM | ~0.1 ms |
| LLM TTFT | Hermes-3-3B FP32 ONNX (CPU) | ~2–10 s (measured, M-series) |
| LLM Decode | — | ~1–3 tok/s (measured, FP32 CPU) |
| TTS First Chunk | Kokoro-82M ONNX | ~10 ms |
| Resampling | SoXR 24k → 16k | ~0.5 ms |
| **TOTAL (First Audio)** | | **seconds, not milliseconds** |

> Dev is a correctness harness, not a latency target. Everything except the LLM
> row is already at prod speed. If you need dev to feel interactive, keep
> `LLM_MAX_NEW_TOKENS` small (default 96) or point `LLM_BACKEND=vllm` at a GPU box.

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
| **TOTAL (First Audio)** | | **~85–135 ms** |

---

## Key Source Files

| File | Purpose |
|------|---------|
| `szca_media_gateway/src/` | All Rust source |
| `src/stage_pool.rs` | Generic `StagePool<R>` core — queue, workers, cancel, backpressure |
| `src/stage_pools.rs` | Pool type aliases, adapters, `StagePools::from_env()`, `SttBackend` + `LlmBackend` enums |
| `src/rt_stt.rs` | Parakeet STT — 3-graph pipeline, TDT decode, Replica impl |
| `src/rt_stt_eou.rs` | Streaming STT — cache-aware Parakeet EOU 120M, chunked encoder cache, RNN-T greedy decode, `<EOU>` detect |
| `src/rt_stt_zipformer.rs` | Streaming STT — Sherpa Zipformer 19-layer, Kaldi 80-mel, 116 cache tensors, stateless decoder, separate joiner |
| `src/rt_stt_mel.rs` | Streaming log-mel frontend — 128 mel, raw `ln(x + 2⁻²⁴)`, **no normalization** |
| `src/rt_llm.rs` | In-process ONNX LLM — KV-cache decode, checkpoint-detected chat template + stop tokens |
| `src/rt_llm_client.rs` | vLLM streaming HTTP client, SSE parser, Replica impl |
| `src/rt_tts.rs` | Kokoro TTS — Misaki G2P, voice packs, sentence-chunked streaming |
| `src/rt_pipeline.rs` | `SttStage` / `LlmStage` / `TtsStage` traits + stubs |
| `src/rt_session.rs` | WS session loop — VAD → pools → barge-in → sentence interleaving; single `run_turn` |
| `src/rt_events.rs` | `ClientCommand` / `ServerEvent` neutral event model |
| `src/rt_protocol.rs` | OpenAI + Gemini dialect adapters |
| `src/api_routes.rs` | HTTP `/v1/stt|llm|tts/stream` + `/v1/pools` + `/metrics` |
| `src/main.rs` | Startup, pool init, graceful shutdown, signal handling |
| `src/vad.rs` + `src/silero.rs` | Silero VAD v5 + RMS fallback |
| `src/dfn3.rs` | DeepFilterNet3 noise cancellation (loaded, wired STFT layer missing) |
| `src/dsp.rs` | Audio DSP utilities |
| `src/onnx.rs` | ORT initialization (load-dynamic) |
| `tests/e2e_pipeline.rs` | E2E turn contract: event order, interleaving, barge-in, dialects, VAD (no weights) |
| `tests/llm_real_inference.rs` | Real dev-LLM: streaming, EOS stop, cancel (skips without weights) |
| `tests/stt_eou_real_inference.rs` | Real streaming EOU STT: decodes `"hello world"` direct + pooled + 20 ms-frame-fed, RTF, silence (skips without weights) |
| `tests/stt_zipformer_real_inference.rs` | Real Zipformer STT: decodes `"Hello World"` direct + pooled + 20 ms-frame-fed, RTF, silence (skips without weights) |
| `tests/silero_real_inference.rs` | Real Silero VAD inference (skips without weights) |
| `tests/dfn3_real_inference.rs` | Real DFN3 3-stage chain (skips without weights) |
| `download_models.sh` | Hardened model download: pinned refs, SHA-256 verify, fails loudly |
| `DEVELOPMENT.md` | Developer quickstart (< 5 min to running) |
| `PROJECT.md` | Single source of truth — architecture, deployment, testing, backlog |

---

## Commands

```bash
# Build
cargo build --manifest-path szca_media_gateway/Cargo.toml
cargo build --release --manifest-path szca_media_gateway/Cargo.toml

# Test (212 unit + 25 integration + 1 doc test, no weights needed)
cargo test --manifest-path szca_media_gateway/Cargo.toml

# Full build check (must be zero warnings)
cargo check --manifest-path szca_media_gateway/Cargo.toml --all-targets

# Run dev
cp env.dev.example .env.dev              # edit ORT_DYLIB_PATH for your OS
set -a && . ./.env.dev && set +a
cargo run --manifest-path szca_media_gateway/Cargo.toml

# Docker dev
docker compose -f docker-compose.dev.yml up --build gateway

# Real-weights tests (opt-in, skip green without env vars)
LLM_MODEL_DIR=$PWD/models/llm/Hermes-3-Llama-3.2-3B \
SILERO_VAD_MODEL=$PWD/models/vad/silero_vad.onnx \
cargo test --release --manifest-path szca_media_gateway/Cargo.toml \
  --test llm_real_inference --test silero_real_inference -- --nocapture

# Download models
./download_models.sh                          # default: Qwen2.5 (1.5 GB LLM, fast dev loop)
LLM_MODEL=hermes3 ./download_models.sh        # Hermes-3-3B (14.4 GB, model-family parity)
```

---

## Key Design Decisions

- **Chat template comes from the checkpoint, never from `model_type`:** Hermes-3 reports `"llama"` but is ChatML-tuned. We read the model's own Jinja `chat_template` from `tokenizer_config.json` and probe which special tokens it emits. A `model_type` heuristic would silently send wrong prompt tokens — degraded replies with no error.
- **Stop tokens are a union, not a priority order:** Hermes-3's `generation_config.eos_token_id` inherits from Llama base (`128001/128008/128009`) while `config.eos_token_id` is `128039` (`<|im_end|>`) — the real stop token. "generation_config wins" drops it and every reply runs to the token cap.
- **One `run_turn` for production and tests:** the per-turn event contract has a SINGLE implementation, driven with pool adapters in production and stub stages in tests. A test that reimplemented the sequence would pass while production drifted.
- **TTS runs in parallel with LLM** (LLM‖TTS overlap): a scoped thread generates tokens and pushes complete sentences through a crossbeam channel while the calling thread reads the channel and runs TTS concurrently. TTS starts before LLM finishes and both make progress simultaneously — the LLM generates sentence N+1 while TTS synthesizes sentence N. Uses `std::thread::scope` and a crossbeam unbounded channel in `run_turn()`.
- **DFN3 noise cancellation is NOT wired yet:** the model loads and the health endpoint confirms it exists, but DFN3 produces a frequency-domain mask at 48 kHz, and the STFT/iSTFT DSP layer to reconstruct enhanced audio hasn't been built yet. Audio flows mic → VAD → STT directly.
- **ORT version coupling:** `ort 2.0.0-rc.10` → `ort-sys::ORT_API_VERSION = 22` → ONNX Runtime 1.22.x. A mismatched dylib makes `ort` refuse to initialize.
- **Streaming STT uses the FP16 EOU export, never the INT8 one:** the INT8 export is built from `ConvInteger`/`MatMulInteger`, and ONNX Runtime has NO CPU kernel for signed-INT8 `ConvInteger` before **1.24** — verified failing on 1.19/1.22/1.23 on both arm64 and x86_64. Switching exports was what made streaming STT possible without touching our `ort` pin. Don't "optimize" to the smaller INT8 file.
- **Streaming mel gets NO normalization:** `rt_stt_mel.rs` produces raw `ln(mel + 2⁻²⁴)`. Adding per-feature or global mean/variance normalization (which the full-utterance `nemo128.onnx` graph does apply) makes the EOU decoder emit **zero tokens** — an empty transcript with no error. A `1e-5` log guard instead of `2⁻²⁴` truncates `"hello world"` to `"hello"`. Both are pinned by tests against golden vectors.
- **`decoder_joint` output: read the LAST logit slot, not slot 0:** it returns `[1, 1, target_plus_sos, 1027]` where slot 0 is the SOS position. Reading slot 0 decodes `"he wor worww"` instead of `"hello world"` — plausible-looking output, not an error.
- **`STT_BACKEND` fails closed:** only `streaming`/`eou` select the EOU model; every other value including typos loads the default Parakeet TDT. `GET /health` reports the RESOLVED backend (`stt_backend`), which is the only signal that a misspelt value silently loaded the wrong model.
- **Replica `process()` runs blocking threads (not async):** each replica is one OS thread, not a tokio task. Session code calls `try_submit_with_cancel` which returns a `Handle` with `.deltas` (tokio mpsc receiver, `blocking_recv()`) and `.done`.
- **Cargo integration tests run with CWD = the package dir:** `szca_media_gateway/` — repo-root-relative paths resolve one level too deep. All real-weights test harnesses use `resolve_from_repo_root()` to handle this.

---

## Key Env Vars

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_BACKEND` | `onnx` | `onnx` (in-process ORT) or `vllm`/`tgi` (external) |
| `LLM_MODEL_DIR` | `./models/llm` | Directory holding `model.onnx` + tokenizer + configs |
| `STT_BACKEND` | `parakeet` | `parakeet` (full-utterance TDT 0.6B), `streaming`/`eou` (cache-aware EOU 120M), or `zipformer` (Sherpa Zipformer). Any other value → `parakeet` |
| `STT_MODEL_DIR` | `./models/stt` | Parakeet model directory |
| `STT_EOU_MODEL_DIR` | `./models/stt_eou` | Streaming EOU model directory (streaming backend only) |
| `SHERPA_MODEL_DIR` | `./models/sherpa_zipformer` | Sherpa Zipformer model directory (zipformer backend only) |
| `TTS_MODEL_DIR` | `./models/tts` | Kokoro model directory |
| `TTS_VOICE` | `af_heart` | Kokoro voice pack name |
| `STT_REPLICAS` | 1 | Parakeet worker count (0=disabled) |
| `LLM_REPLICAS` | 1 | LLM worker count (0=disabled) |
| `TTS_REPLICAS` | 1 | Kokoro worker count (0=disabled) |
| `SILERO_VAD_MODEL` | (empty → RMS) | Path to Silero VAD ONNX model |
| `ORT_DYLIB_PATH` | auto-detect | Path to `libonnxruntime.so/dylib` |
| `SZCA_LISTEN_ADDR` | `0.0.0.0` | Bind address |
| `SZCA_PORT` | 3000 | Listen port |
| `SZCA_MAX_SESSIONS` | 1000 | Admission cap (excess → 503) |
| `SZCA_API_KEY` | (disabled) | Bearer token for API auth |
| `SZCA_QUEUE_BACKLOG` | 64 | Per-pool queue backlog; set to 1024+ for prod at 1k sessions |
| `SZCA_WORKER_THREADS` | CPU count | Tokio async worker threads (default = CPUs). Set 32+ for 1k sessions |
| `SZCA_BLOCKING_THREADS` | 512 | Tokio blocking thread pool max. 256+ for 200 concurrent turns |
| `RUST_LOG` | (none) | Log level: `info`, `debug`, `trace` |
| `LLM_MAX_NEW_TOKENS` | 256 | Per-turn token cap (onnx backend only) |
| `LLM_CHAT_TEMPLATE` | auto-detect | Force `chatml` or `llama` (onnx backend only) |

Ready-to-copy templates: `env.dev.example` and `env.prod.example` (`.env*` is gitignored).

---

## Key Facts to Remember

- Models live under `models/{stt,tts,llm,vad,dfn3}` — one root, subdirs per stage
- All weights are gitignored; download with `./download_models.sh` (~16 GB total for Hermes-3 profile)
- `LLM_MODEL` is OVERLOADED: in `download_models.sh` it selects which model to download; in the gateway it is the API model name sent to vLLM (ignored when `LLM_BACKEND=onnx`)
- The gateway's built-in defaults (`./models/{stt,tts,llm}`) let you `cargo run` from repo root with no path env vars
- The INT8 dev LLM (Qwen2.5-1.5B, ~1.5 GB) loads in seconds and runs at ~10-20 tok/s on CPU. Hermes-3-3B FP32 (~14.4 GB, ~40s load, ~1-3 tok/s) is available via `LLM_MODEL=hermes3` for model-family testing.
- Test status: 213 unit + 25 integration + 1 doc = 239 tests, zero warnings
- Noise cancellation (DFN3) is fully wired into `rt_session.rs` via `dfn3_dsp.rs` and `deep_filter` crate for 48kHz STFT/iSTFT analysis & synthesis.
- `Dockerfile.dev` uses `cargo-watch` for hot-reload with dep caching; models are bind-mounted readonly

- **Per-pool latency histograms** are tracked via `hdrhistogram`. Each worker records wall-clock `Replica::process` duration (ms) into an `Arc<Mutex<Histogram>>`. The `/v1/pools` endpoint exposes `{stt,llm,tts}_latency` with `count`, `min_ms`, `max_ms`, `mean_ms`, `p50_ms`, `p90_ms`, `p95_ms`, `p99_ms` per stage. Returns `null` until ≥2 jobs have completed.
