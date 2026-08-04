# SZCA Media Gateway — Action Items & TODOs

All verified issues discovered during code review, organized by priority.

**Status check:** 2026-07-31 — open items re-verified against current source. Several earlier “[x]” items were reopened where the live path still lacks the fix.

---

## 🔴 Critical Priority — Must Fix Before Production

### From earlier review (resolved / verified)

- [x] **Half-open WebSocket hang (resource leak):** Fixed — 35s idle read timeout + outbound `Message::Ping` every 30s so half-open connections are probed and eventually drained.
- [x] **`protocol.rs` / dialect deserialization panics:** Production dialect decode in `rt_protocol.rs` uses `match` / soft-fail (empty cmds), not unwrap-on-untrusted-input. Test-only unwraps remain.
- [x] **VAD EOU threshold mismatch:** Default `speech_threshold` is **0.35** in `vad.rs` (verified).
- [x] **Silero model loaded per-session (cold-start latency):** False alarm — Silero loads via `vad.rs` + `silero.rs` per `VadProcessor`, not in `rt_stt_eou`.
- [x] **`process_pcm_f32()` allocates on every frame:** Fixed — pre-allocated `scratch` buffer in `ParakeetEouStt`.
- [x] **`tokio::spawn` with dropped JoinHandle (orphan tasks):** Session awaits the outbound JoinHandle and uses a broadcast cancel token on teardown.

### Reopened — marked done earlier, but NOT done on the live path

- [x] **No backpressure on audio pipeline (still incomplete):** Fixed — removed unused `_vad_to_stt_*` bounded channel. Real caps: (1) `MAX_UTTERANCE_BYTES` (30 s PCM) with oldest-drop on overflow in `append_utterance`; (2) StagePool `SZCA_QUEUE_BACKLOG` for inference. Full async VAD→STT channel wiring deferred until streaming STT is pool-backed.
- [x] **VAD filtering clips speech onset (lookback not wired):** Fixed — rolling `LOOKBACK_BYTES` (~200 ms) ring; on `SpeechStart`, prepends to utterance and feeds streaming STT when `supports_lookback()` is true (EOU + Zipformer).
- [x] **No audio input validation on WS path:** Fixed — `validate_pcm16_chunk` on `AppendAudio` (even-byte PCM16, 1 MiB cap); rejects with `ServerEvent::Error`.

### Open — from full codebase review (2026-07-31)

- [ ] **WebSocket `/v1/realtime` has no API-key auth:** HTTP routes enforce `SZCA_API_KEY`; WS only does admission (503). **Deferred** until security pass (see Security backlog). Acceptable while localhost-only or edge-enforced.
- [x] **Per-session streaming STT bypasses StagePool:** Fixed — sessions no longer call `create_real_stt()`. Streaming/Zipformer use a shared `StreamingSttPool` free-list lease (`try_acquire_streaming_stt`); batch Parakeet stays on the StagePool job queue. Concurrent early-EOU sessions capped at `STT_REPLICAS`; excess fall back to VAD + full-utterance lease/queue transcribe.
- [x] **`szca_tests` CI does not exercise the real gateway:** Renamed CI jobs to "Mock Integration Patterns", "Reference Security Logic", "Mock Performance Benchmarks"; `szca_tests/lib.rs` documents scope. Real coverage remains `szca_media_gateway/tests/e2e_pipeline.rs` in `rust-tests`.

---

## 🟠 High Priority — Production Hardening

### From earlier review (resolved / verified)

- [x] **Code duplication: `vad.rs` vs `rt_stt_eou.rs`:** No VAD impl in `rt_stt_eou` — comments only.
- [x] **Misleading log levels:** Session teardown uses `info!`.
- [x] **Unclosed response writer on error path:** No fixable issue under axum.
- [x] **`docker-compose.prod.yml` uses `network_mode: host`:** Replaced with `szca-net`.
- [x] **`Dockerfile` uses `curl ... | bash`:** Not present; SHA-256 downloads.
- [x] **SecComp / AppArmor disabled in production Compose:** `no-new-privileges` + `cap_drop: ALL`.
- [x] **vLLM workers share `model_cache` volume RW:** Not in current compose.
- [x] **CPU oversubscription on non-GPU instances:** Gateway has CPU/memory limits.
- [x] **Release profile has debug symbols:** `strip = true` in release.
- [x] **No CORS on direct Redis exposure:** Internal `expose` + password; no host port.

### Open — from full codebase review (2026-07-31)

- [ ] **Docker nginx TLS incomplete:** Compose maps 443; `nginx.conf` only listens on 80. Helm/ALB TLS is fine. **Fix:** TLS block + certs, or document ALB-only. (Also listed under Security backlog.)
- [x] **`POOL_QUEUE_CAP` vs `SZCA_QUEUE_BACKLOG` mismatch:** Fixed — `api_routes::acquire_pool` now uses `stage_pool::queue_backlog_from_env()` (same as `StagePool::build`). Hardcoded `POOL_QUEUE_CAP = 64` removed.
- [x] **Dead / unused backpressure channel in session loop:** Removed `_vad_to_stt_*` / `MAX_PIPELINE_DEPTH`; replaced with `MAX_UTTERANCE_BYTES` cap via `append_utterance`.
- [x] **Heartbeat branch is a no-op:** Fixed — outbound task sends `Message::Ping` every 30s; idle still enforced by 35s read timeout. No-op watch channel / heartbeat task removed.
- [x] **Legacy modules still compiled (~1.9k LOC):** Removed dead `ipc`, `protocol`, `ring_buffer`, legacy `gateway`/`session` paths. Live config → `config.rs`; admission → `admission.rs`. Production path is StagePool in-process (not SHM IPC).

---

## 🟡 Moderate Priority — Reliability & Correctness

### From earlier review (resolved / verified)

- [x] **Silero EOU context size mismatch:** `SILERO_WINDOW_SAMPLES = 512` — correct for v5 @ 16 kHz.
- [x] **No negative tests for VAD edge cases:** Covered in `vad.rs` unit tests (weights-dependent cases out of scope).
- [x] **No negative tests for protocol parser:** Fuzz-style tests present in `rt_protocol.rs`.
- [x] **`DEVELOPMENT.md` QEMU workaround:** Not present.
- [x] **No CI pipeline defined:** `.github/workflows/ci.yml` exists with full job set.

### Open — from full codebase review (2026-07-31)

- [ ] **`/health` always public and informative:** Leaks version, STT backend, model KiB. Product decision: minimal public vs rich internal. (Security backlog.)
- [ ] **CI action pins are mutable tags:** Pin `uses:` to commit SHAs.
- [x] **`rt_pipeline.rs` header still says Phase-1 stubs:** Updated — documents production ONNX stages + StagePools.

---

## 🔒 Security backlog (deferred — add later)

Agreed approach for now:
- **Dev:** `SZCA_LISTEN_ADDR=127.0.0.1`, no API key, plain HTTP/WS.
- **Prod:** TLS at edge (ALB/nginx), `SZCA_API_KEY` on HTTP `/v1/*` + `/metrics`; gateway plain HTTP on trusted network.
- Do **not** expose gateway/WS directly to the public internet until WS auth lands.

When security work resumes:

- [ ] Bearer auth on `/v1/realtime` WebSocket upgrade (same `SZCA_API_KEY` as HTTP)
- [ ] Confirm real TLS on Docker nginx path (or ALB-only + document)
- [ ] Optional lockdown / split of `/health` and `/v1/pools` (public vs internal)
- [ ] Rate limiting / abuse controls on WS session creation
- [ ] API key rotation story (secret manager, no keys in compose env files)
- [ ] Constant-time token compare if timing side-channels matter
- [x] **`/metrics` behind same bearer when key set:** Already wired via `api_router` + `require_auth` (verified).

---

## 🔵 Future Roadmap (Phase 2+)

### Load Testing & Validation (from original TODO)

- [ ] **1000-session WebSocket load test:** Execute `locustfile.py` and `szca_load_test/ws_load_test.js` on GPU hardware (`g6e.48xlarge`) to validate `SZCA_QUEUE_BACKLOG` sizing, GPU saturation, and overall system stability.
  - BLOCKER: Requires AWS `g6e.48xlarge` provisioning by DevOps.

### Upstream Tech Debt (from original TODO)

- [ ] **ort Upgrade:** Upgrade `ort` from `=2.0.0-rc.10` to `2.0.0` stable once published.
  - BLOCKER: `ort 2.0.0` not yet released to crates.io.
- [ ] **Spanish TTS Support:** Integrate a non-English G2P engine to replace/augment `misaki-rs` (English-only).
- [ ] **Hermes-3-3B INT8:** Explore local Hermes-3-3B INT8 for improved local reasoning.
  - BLOCKER: Requires 32GB+ RAM for `optimum-cli` quantization.

### Architecture / cleanup (from full review)

- [ ] Clarify or remove C++ `szca_onnx_engine` from primary CI if Rust gateway fully superseded it
- [ ] Doc pass: align `CLAUDE.md` / `PROJECT.md` “zero-copy IPC” claims with in-process StagePool reality

---

## ✅ Completed (from original TODO)

- [X] Production Multi-Tier Docker Compose (`docker-compose.prod.yml`)
- [X] Nginx Reverse Proxy & Load Balancer (`deploy/nginx/nginx.conf`)
- [X] vLLM Healthcheck `start_period: 360s`
- [X] Locust Load Test Endpoint (`locustfile.py`)
- [X] Chaos Test Suite (`scripts/chaos_test.py`)
- [X] STT/TTS Replica Sizing & Backend Documentation

---

## Recommended next fix (non-security)

Security is deferred. Suggested order:

1. ~~**`POOL_QUEUE_CAP` → use `SZCA_QUEUE_BACKLOG`**~~ ✅ Done.
2. ~~**Session cleanup** (dead channel + real WS pings)~~ ✅ Done.
3. ~~**Per-session streaming STT → StagePool / lease pool**~~ ✅ Done.
4. ~~**Audio validation + VAD lookback**~~ ✅ Done.
5. ~~**Trivial:** rewrite `rt_pipeline.rs` Phase-1 header~~ ✅ Done.
