# Work Summary — SZCA Realtime Voice Engine

**Project:** SZCA Realtime Voice Engine

---

## July 31, 2026 — Codebase Review, Production Hardening & CI Honesty

**Context:** Full structured codebase review; implement non-security fixes in priority order; remove legacy Phase-0 modules; make CI job names honest about what `szca_tests` actually runs.

### What We Did

#### 1. Review & tracking (`todo.md`)

- Audited the gateway against production readiness (session lifecycle, backpressure, auth gaps, pool wiring, legacy code).
- Re-verified open items against live source; reopened three items that were falsely marked done (backpressure, VAD lookback, audio validation).
- Marked completed fixes as done; deferred WebSocket auth and nginx TLS to the security backlog (dev = localhost, prod = TLS at edge + HTTP bearer).

#### 2. Queue backlog parity

- **`stage_pool.rs`:** exported `DEFAULT_QUEUE_BACKLOG`, `queue_backlog_from_env()`, `parse_queue_backlog()`.
- **`api_routes.rs`:** removed hardcoded `POOL_QUEUE_CAP = 64`; `acquire_pool` now uses the same env-driven backlog as `StagePool::build`.

#### 3. Session hardening (`rt_session.rs`)

- Removed dead `_vad_to_stt_*` bounded channel / `MAX_PIPELINE_DEPTH` (never wired).
- Added `MAX_UTTERANCE_BYTES` (30 s PCM) with oldest-drop via `append_utterance()`.
- Outbound task sends real `Message::Ping` every 30 s; removed no-op heartbeat watch channel.
- Fixed `Hangup` to `break 'session` (was not exiting the loop).
- Unit tests for utterance cap.

#### 4. Streaming STT → shared pool (no per-session `create_real_stt`)

- **`stage_pools.rs`:** `StreamingSttPool` + free-list lease for sticky `push_chunk` state; `SttPoolAdapter` enum (`Queued` vs `Leased`).
- **`rt_session.rs`:** uses `pools.try_acquire_streaming_stt()`; concurrent early-EOU sessions capped at `STT_REPLICAS`.
- **`api_routes.rs`:** STT HTTP path and `/v1/pools` report lease `in_use` for streaming backends.

#### 5. Audio validation + VAD lookback

- **`rt_pipeline.rs`:** `validate_pcm16_chunk()`, `MAX_AUDIO_CHUNK_BYTES`; module header updated (no longer “Phase-1 stubs only”).
- **`rt_session.rs`:** validates `AppendAudio`; rejects with `ServerEvent::Error`.
- VAD lookback: ~200 ms rolling ring; on `SpeechStart`, prepends to utterance and feeds streaming STT when `supports_lookback()` (EOU + Zipformer).

#### 6. Legacy module cleanup (~60k LOC removed)

Production path is in-process **StagePool** — not SHM IPC. Removed dead Phase-0 code:

| Removed | Replacement |
|---------|-------------|
| `gateway.rs`, `protocol.rs` | `config.rs` — `GatewayConfig::from_env()` |
| `session.rs` (legacy `Session` state machine) | `admission.rs` — `SessionManager` only |
| `ipc.rs`, `ring_buffer.rs` | Deleted (never on live path) |

`main.rs` imports `config::GatewayConfig` and `admission::SessionManager`; stale “zero-copy IPC” comment updated.

#### 7. CI honesty (`szca_tests`)

`szca_tests` has **no** dependency on `szca_media_gateway` — it runs mock pipelines and reference helpers. Real gateway contract tests live in `szca_media_gateway/tests/e2e_pipeline.rs` (Rust Gateway Tests job).

| Old CI job name | New job name |
|-----------------|--------------|
| Integration Tests | **Mock Integration Patterns (szca_tests)** |
| Security Tests | **Reference Security Logic (szca_tests)** |
| Performance Benchmarks | **Mock Performance Benchmarks (szca_tests)** |

`szca_tests/lib.rs` documents scope in a table (mocks vs real gateway coverage).

### Test status (July 31)

| Crate | Result |
|-------|--------|
| `szca_media_gateway` | **141 tests pass** (unit + integration; legacy module tests removed with deleted code) |
| `szca_tests` | **154 tests pass** (mocks/reference logic) |

Real-weights tests remain opt-in (`stt_eou_real_inference`, `llm_real_inference`, etc.).

### Still open (non-security)

- Docker nginx TLS incomplete (443 mapped, nginx listens on 80 only).
- `/health` info leak (product decision).
- CI action SHA pins (mutable tags).
- Doc pass: align `CLAUDE.md` / `PROJECT.md` “zero-copy IPC” claims with StagePool reality.
- C++ `szca_onnx_engine` CI relevance.

### Security backlog (deferred)

- Bearer auth on `/v1/realtime` WebSocket.
- nginx TLS or ALB-only documentation.
- Optional `/health` / `/v1/pools` split; WS rate limiting; key rotation.

### Files touched (July 31)

**New**

- `szca_media_gateway/src/config.rs`
- `szca_media_gateway/src/admission.rs`

**Deleted**

- `szca_media_gateway/src/gateway.rs`
- `szca_media_gateway/src/protocol.rs`
- `szca_media_gateway/src/ipc.rs`
- `szca_media_gateway/src/ring_buffer.rs`
- `szca_media_gateway/src/session.rs`

**Modified**

- `szca_media_gateway/src/lib.rs`, `main.rs`, `rt_session.rs`, `rt_pipeline.rs`
- `szca_media_gateway/src/stage_pool.rs`, `stage_pools.rs`, `api_routes.rs`
- `szca_media_gateway/src/rt_stt_eou.rs`, `rt_stt_zipformer.rs` (`supports_lookback`)
- `.github/workflows/ci.yml`
- `szca_tests/lib.rs`
- `todo.md`

---

## July 26, 2026 — Streaming STT Implementation

**Context:** Build and ship streaming STT as an optional, configurable alternative to the default full-utterance Parakeet TDT 0.6B. Evaluate all three models on accuracy and pick the default.

### What We Built

#### Three STT Models, One Pipeline

All three share the same `SttPoolAdapter`, `StagePool`, and session code through `SttBackend` enum dispatch. Selectable via `STT_BACKEND=parakeet|streaming|eou|zipformer`. Fails closed — any unrecognized value loads the default Parakeet TDT. `GET /health` reports the resolved backend.

#### New Source Files

| File | Line Count | Purpose |
|---|---|---|
| `szca_media_gateway/src/rt_stt_mel.rs` | ~350 | 128-mel raw log-mel frontend, **no normalization** |
| `szca_media_gateway/src/rt_stt_eou.rs` | ~450 | Parakeet EOU 120M cache-aware stage, 4 encoder caches, `<EOU>` detect |
| `szca_media_gateway/src/rt_stt_zipformer.rs` | ~490 | Sherpa Zipformer 19-layer stage, Kaldi 80-mel, 116 cache tensors, stateless decoder |

#### Real-Weights Tests

| Test File | Tests | Verified |
|---|---|---|
| `tests/stt_eou_real_inference.rs` | 7 | "hello world" direct, pooled, 20ms-frame; RTF; silence; reset |
| `tests/stt_zipformer_real_inference.rs` | 6 | "Hello World" direct, pooled, 20ms-frame; RTF; silence; reset |

#### Wiring Changes

- `szca_media_gateway/src/stage_pools.rs` — `SttBackend` enum, `dev_model_selection()`, `zipformer_stt_selected()`
- `szca_media_gateway/src/rt_pipeline.rs` — `Pipeline::with_real_models()` matches pool backend selection
- `szca_media_gateway/src/main.rs` — `GET /health` reports resolved backend
- `download_models.sh` — step [5/8] with SHA-256 pins for all 4 files

#### CI/Test Status (July 26)

**239 tests, zero warnings:**

- 213 unit + 25 integration + 1 doc-test
- All pass with no weights needed
- `cargo check --all-targets` clean
- No new crate dependencies (only `realfft = "3.5"` promoted from transitive)

### WER Evaluation — 174 LibriSpeech test-clean utterances

| Rank | Model | `STT_BACKEND=` | WER | Avg decode | RTF | Size |
|---|---|---|---|---|---|---|
| 🥇 | **Parakeet TDT 0.6B** | `parakeet` (default) | **3.28%** | 0.93 s | — | ~400 MB |
| 🥈 | **Sherpa Zipformer** | `zipformer` | **19.01%** | 0.41 s | ~22× | 156 MB |
| 🥉 | Parakeet EOU 120M | `streaming` / `eou` | **37.69%** | 0.48 s | ~15× | 236 MB |

**Methodology:** Python ONNX Runtime wrappers that replicate the exact Rust pipelines. `jiwer` with case-insensitive, punctuation-stripped comparison.

**Decision:** TDT 0.6B stays default. Zipformer is the recommended streaming option. EOU only when the `<EOU>` signal is specifically needed.

### Silent Failure Modes (All Test-Guarded)

| Mistake | Symptom | Which model |
|---|---|---|
| Any mel normalization | `""` — zero tokens, no error | EOU |
| Log guard `1e-5` instead of `2⁻²⁴` | `"hello"` — truncated | EOU |
| Joint logit slot 0 instead of last | `"he wor worww"` — plausible garbage | EOU |
| Zero-dimension cache tensors | Encoder fails at runtime | Zipformer |
| Pair-format vocab parser on line-index vocab | `""` — empty vocab, no error | EOU |
| Caches not carried between chunks | Every chunk decoded fresh | Both |
| `reset()` missing between pool jobs | Turn N+1 inherits N's context | Both |

### Key Design Decisions (Recorded in CLAUDE.md)

1. **FP16 EOU export, never INT8.** The INT8 export uses `ConvInteger`/`MatMulInteger`; ONNX Runtime has no CPU kernel for signed-INT8 `ConvInteger` before **1.24**. We run ORT 1.22. The FP16 export runs at 15× realtime.
2. **Streaming mel gets NO normalization.** Adding per-feature or global mean/variance normalization makes the EOU decoder emit zero tokens — verified empirically.
3. **`decoder_joint` output: read the LAST logit slot.** Slot 0 is the SOS position; reading it decodes plausible garbage ("he wor worww") instead of correct text ("hello world").
4. **`STT_BACKEND` fails closed.** Only `streaming`/`eou`/`zipformer` select streaming models; any typo silently loads the default TDT. `/health` reports the resolved choice.
5. **No `ort` upgrade needed.** The entire streaming STT implementation uses existing `ort 2.0.0-rc.10` / ONNX Runtime 1.22. No `Cargo.toml` changes beyond `realfft`.
6. **Cache tensors must NOT be zero-length.** `ort` rejects zero-dimension tensors — all 116 Zipformer cache tensors must be instantiated with their correct ONNX-specified shapes.

### Completed Follow-Up Tasks (July 26)

- **Priority 1:** Wire `<EOU>` as turn-end trigger in `rt_session.rs` ✅
- **Priority 2:** Pin ORT in prod Dockerfile ✅
- **Priority 3:** Gate EOU/Zipformer download behind `--with-streaming` ✅

### Files Modified / Created (July 26)

**New:** `rt_stt_mel.rs`, `rt_stt_eou.rs`, `rt_stt_zipformer.rs`, `tests/stt_eou_real_inference.rs`, `tests/stt_zipformer_real_inference.rs`

**Modified:** `Cargo.toml`, `lib.rs`, `stage_pools.rs`, `rt_pipeline.rs`, `main.rs`, `download_models.sh`, `env.dev.example`, `env.prod.example`, `PROJECT.md`, `CLAUDE.md`, `DEVELOPMENT.md`
