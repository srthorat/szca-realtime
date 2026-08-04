# Scaling Plan — ARCHIVE (Historical Reference)

> ⚠️ **This document is archived.** See `PROJECT.md` for the current single source of truth.
> Kept for "why did we choose this?" decision reasoning and phase-by-phase evolution.

_Created: 2026-07-23 · Architecture locked: 2026-07-23_

## LOCKED ARCHITECTURE (decision of record)

Ultrascalable voice = **our Rust engine owns STT/TTS/VAD/orchestration; vLLM (or
TGI) owns LLM token generation on your GPUs.** We do NOT build our own LLM
batching engine — that is re-implementing vLLM on a runtime (ORT) without the
custom CUDA kernels, months of work, and still slower. We do NOT chase
Cerebras/Groq raw tokens/sec — that is custom silicon, unreachable in software.

For **voice**, perceived "big-player" quality = low time-to-first-audio +
full streaming + fast barge-in (all largely built), NOT raw tok/s. ~30–50 tok/s
already outpaces human listening.

```
OUR RUST ENGINE                              vLLM / TGI on your GPUs
────────────────                             ───────────────────────
• STT pool (Parakeet)  in-process ONNX       • Hermes/Llama generation
• TTS pool (Kokoro)    in-process ONNX       • continuous batching
• VAD / barge-in / turn-taking               • paged KV-cache
• streaming orchestration, dual dialect      • tensor-parallel, 1000s streams
• LLM stage = streaming CLIENT ───────────►  OpenAI-compatible API (stream=true)
```

**NOT our scope** (provided to the engine as requests when needed): flow engine
/ deterministic state machine, RAG pipeline, guardrail/validation layer. The
engine exposes seams these plug into (in-process pool API + WS/HTTP endpoints).

---


## Goal

Three **independent, composable** inference services — STT, LLM, TTS — that can
be wired into any pipeline:

- `STT-only`, `LLM-only`, `TTS-only`
- `STT+LLM`, `LLM+TTS`
- `STT+LLM+TTS` (full speech-to-speech)

Capacity must be **fungible across stages**, drawn from one hardware budget. A
box that can do **300 full S2S streams** must instead be reallocatable — by
config, same binary — as any of:

- 100 STT-only + 100 LLM-only + 100 TTS-only
- 100 STT-only + 200 (LLM+TTS)
- any split where `Σ (replicas_per_stage × cost_per_replica) ≤ hardware`

This is a cheap alternative to Gemini Live / OpenAI Realtime, trading absolute
per-stream latency for density and horizontal composability.

---

## Why the current architecture fails this

Two orchestration facts (the model code itself is fine and fully reused):

1. **Model-per-session** — `Pipeline::with_real_models()` is called *inside*
   `run_session`, so every WebSocket connection loads its own full copy of every
   model. 100 sessions ⇒ ~100× weights in RAM (~250 GB). Fatal.
2. **Whole-turn mutex** — the session holds `Mutex<Pipeline>` for the entire
   STT→LLM→TTS turn, and ONNX `Session::run(&mut self)` is `&mut`, so inference
   is serialized. Effective concurrency ≈ 1.

Neither is a model-code problem; both are wiring. The fix is a **shared
replica-pool + queue** layer between sessions and models.

---

## Target architecture

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

- **Replica** = one loaded model instance owning its ONNX `Session`, running on a
  dedicated blocking thread, pulling jobs off the pool's queue.
- **Job** = `{ input, cancel: Arc<AtomicBool>, result_tx }`. The worker streams
  deltas (transcript / token / audio) back over `result_tx`; the caller awaits
  them. Barge-in cancels via the shared flag exactly as today.
- **Pool** = `{ job_tx, permits }`. `submit(job)` enqueues; concurrency is
  bounded by replica count (natural backpressure — queue depth is observable).
- **Sessions own no models.** They hold `Arc<StagePools>` and submit to the
  pools their composition requires. WS realtime and HTTP endpoints share the
  same pools.

### Fungibility

`N_stt`, `N_llm`, `N_tts` are independent env config. A stage with `N = 0` is
disabled (its pool isn't built, its models never load). So:

| Deployment | Config |
|---|---|
| Full S2S ×K | `STT=K LLM=K TTS=K` |
| STT-only ×100 | `STT=100 LLM=0 TTS=0` |
| 100 STT + 200 (LLM+TTS) | `STT=100 LLM=200 TTS=200` |

Admission control becomes **per-stage** (a permit/semaphore per pool) instead of
one global session cap, so a full-pipeline session acquires one permit from each
of the three pools it uses.

---

## Phases

### Phase 1 — Stage-pool core (foundation) ✅ DONE
- `stage_pool.rs`: generic `StagePool<R: Replica>` over crossbeam-channel bounded queue.
- `Job` type carrying input + `Arc<AtomicBool>` cancel flag + streaming delta/finish channels.
- Worker loop: one blocking thread per replica, streams deltas back.
- `try_submit` (non-blocking) + `try_submit_with_cancel` (caller's own cancel Arc).
- `queue_depth()` AtomicUsize counter, `replica_count()` accessor.
- Unit tests: concurrency, cancel, backpressure, queue full.

### Phase 2a — Wrap STT + TTS + LLM as pools ✅ DONE
- `Replica` impls for `ParakeetStt`, `KokoroTts`, `QwenLlm` + input types.
- `stage_pools.rs`: `SttPool`, `LlmPool`, `TtsPool` type aliases + pool adapters.
- `StagePools` struct with `from_env()`: models load once at startup.
- `LlmBackend` enum: config-driven `onnx` (in-process) or `vllm` (external).
- Env config: `STT_REPLICAS`, `LLM_REPLICAS`, `TTS_REPLICAS` (0 = disabled).

### Phase 2b — vLLM streaming client ✅ DONE
- `rt_llm_client.rs`: `VllmClient` implementing `Replica` trait.
- Streaming HTTP to `/v1/chat/completions` with SSE token delta parsing.
- Config: `LLM_BASE_URL`, `LLM_MODEL`, `LLM_API_KEY`, `LLM_MAX_TOKENS`, `LLM_TEMPERATURE`.
- Tokio runtime handle captured at creation (pool workers are plain OS threads).

### Phase 3 — Rewire WS sessions onto pools ✅ DONE
- `run_session` takes `Option<Arc<StagePools>>` — no per-session model copies.
- `run_response` creates pool adapters per turn; sentence-chunked interleaving preserved.
- `Mutex<Pipeline>` eliminated — pool adapters are separate structs (split borrows work).
- Barge-in cancel propagates via `try_submit_with_cancel` to the same `AtomicBool`.

### Phase 4 — Rewire HTTP endpoints onto pools ✅ DONE
- `/v1/stt/stream`, `/v1/llm/stream`, `/v1/tts/stream` on shared pools.
- `spawn_blocking` for pool work, SSE streaming via `futures::stream`.
- Request validation, bearer token auth, body size limits.
- `/v1/pools` health endpoint (available, queue_depth, replicas per stage).

### Phase 5 — Metrics, backpressure, health ✅ DONE
- Per-pool `queue_depth()` AtomicUsize counter (increment on submit, decrement on pickup).
- 503 + `Retry-After` when pool queue exceeds `POOL_QUEUE_CAP` (64).
- `/v1/pools` reports per-stage replica count + saturation.
- `/metrics` Prometheus export (pre-existing).

### Phase 6 — Production hardening ✅ DONE
- **Graceful shutdown** (SIGTERM + 30s drain before exit). ✅
- **Request timeout middleware** (tower-http TimeoutLayer, 60s). ✅
- **Load testing** (k6 HTTP endpoints + WebSocket realtime). ✅
- **CI/CD** (GitHub Actions: 10 jobs — test, clippy, fmt, audit, Docker, Trivy, gitleaks). ✅ Pre-existing.
- **Banner → tracing::info!** (no more println!). ✅
- **Fix unwrap()** in main.rs (proper error handling everywhere). ✅

### Phase 7 — Remaining optimizations (not blocking production)
- End-to-end integration test (PCM→STT→LLM→TTS→PCM over live WS).
- Per-pool latency histograms (request duration p50/p95/p99).
- True LLM‖TTS overlap (parallel generation + synthesis on separate threads).
- DFN3 noise cancellation (wire into pipeline, 48 kHz STFT DSP).
- Streaming STT input (cache-aware Parakeet export for incremental audio).
- Spanish TTS (non-English G2P engine).

---

## Deployment Profiles (Two-Track)

| | DEV | PROD |
|--|-----|------|
| LLM backend | `LLM_BACKEND=onnx` (in-process) | `LLM_BACKEND=vllm` (external GPU) |
| LLM model | Qwen2.5-1.5B int8 (~1.5 GB) | Llama 3.1 8B FP8 (~8 GB) |
| STT/TTS | In-process ORT, 1 replica each | In-process ORT, 4–8 replicas each |
| Hardware | CPU (MacBook/Linux VM) | 2× A100 or 3× L40S |
| Concurrency | 1–2 (correctness only) | 300+ (load tested) |
| Docker | `docker-compose.dev.yml` | `docker-compose.prod.yml` |
| Infra | Just `cargo run` | Nginx + Redis + GPU workers |
| API contract | Same WebSocket + HTTP endpoints | Identical (contract tested) |

Switch: `LLM_BACKEND=onnx` → dev. `LLM_BACKEND=vllm` → prod. Zero code changes.
See [WORK_SUMMARY.md](WORK_SUMMARY.md#deployment-profiles) for full spec.

---

## Hardware sizing (rule of thumb, refined in Phase 5)

Per-stage replica cost is measured, not guessed. Expected shape:

| Stage | Per-replica cost | Notes |
|---|---|---|
| STT (Parakeet TDT int8) | moderate CPU, bursty | fires once per turn on `SpeechEnd` |
| LLM (Qwen2.5-1.5B int8) | heaviest; GPU or batching for density | dominant bottleneck for full S2S |
| TTS (Kokoro + Misaki) | light, batches well | pure-Rust G2P, no FFI |

Full-pipeline 100+ concurrent realistically wants GPU + batched LLM (Phase 6) or
an external LLM server; STT-only / TTS-only fleets scale on multi-core CPU.

---

## Invariants preserved through all phases
- Streaming with real interim + final per stage (already built).
- Barge-in via shared `AtomicBool` cancel.
- Dual wire dialects (OpenAI Realtime + Gemini Live).
- Models never committed to git; hardened download + SHA-256 pinning.
- Every phase leaves the tree building clean with all tests green.
```
