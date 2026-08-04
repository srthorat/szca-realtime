# Development Guide

> Get the SZCA Media Gateway running locally in under 5 minutes.

## Prerequisites

| Tool | Version | Install |
|------|---------|---------|
| **Rust** | 1.75+ | `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \| sh` |
| **Docker** | 24+ | [docker.com](https://docs.docker.com/get-docker/) (optional, for containerized dev) |
| **ONNX Runtime** | 2.0+ | Auto-loaded at runtime via `ORT_DYLIB_PATH` (or vendored dylib) |

## Quick Start (Local, No Docker)

```bash
# 1. Clone and enter the repo
git clone <repo-url> && cd realtime

# 2. Download models (choose one):
#    Quick dev (~1.8 GB total) — Qwen2.5-1.5B INT8, loads in seconds:
./download_models.sh
#    Full parity (~16 GB total) — Hermes-3-3B FP32, same family as prod:
# LLM_MODEL=hermes3 ./download_models.sh

# 3. Load the dev profile (ORT dylib path, model dirs, pool sizes)
cp env.dev.example .env.dev
$EDITOR .env.dev                 # ORT_DYLIB_PATH must match your install
set -a && . ./.env.dev && set +a

# 4. Build and run
cargo run --manifest-path szca_media_gateway/Cargo.toml

# 5. Verify
curl http://localhost:3000/health
curl http://localhost:3000/v1/pools    # per-stage replicas, queue depth + latency histograms (p50/p90/p95/p99)
```

The gateway starts on `:3000` with stub STT/LLM/TTS if models aren't found,
or real inference if models + ORT are available. **Stubs are silent by
design** — the only way to know which you got is the startup log:

```
INFO Starting SZCA Media Gateway  addr=127.0.0.1:3000 max_sessions=1000
INFO Building LLM pool (ONNX)  replicas=1
INFO LLM chat template selected  chat_template=ChatML
INFO ONNX causal-LM loaded  n_layers=28 n_kv_heads=8 head_dim=128
     max_new_tokens=96 chat_template=ChatML eos_ids=[128001, 128009, 128039]
INFO All configured inference pools ready
INFO Server ready
```

Three things to check in that output:

| Line | What it proves |
|------|----------------|
| `chat_template=ChatML` | Hermes-3 is ChatML-tuned despite `model_type: "llama"`. `Llama` here means wrong prompt format → degraded replies, **no error**. |
| `eos_ids` contains `128039` | `<|im_end|>` is the real stop token. Without it every reply runs to `LLM_MAX_NEW_TOKENS`. |
| `All configured inference pools ready` | Real models. `Failed to build inference pools; sessions will use stubs` means `LLM_MODEL_DIR` / `ORT_DYLIB_PATH` is wrong — fix that before judging output quality. |

> The first `cargo run` after a Hermes download takes ~40 s to become healthy:
> ORT has to map 14.4 GB of external weights. That cost is per process start,
> not per request.

### Swapping the dev LLM

Qwen2.5-1.5B INT8 is the default dev LLM: ~1.5 GB, self-contained (no external
data file), loads in seconds, ~10-20 tok/s on CPU. Nothing in the code changes
— the KV geometry comes from `config.json` and the prompt format from
`tokenizer_config.json`.

Hermes-3-Llama-3.2-3B FP32 is available for model-family testing (same family
as the prod Hermes-3-Llama-3.1-8B). It is a 14.4 GB export with ~40 s load time
and single-digit tok/s — useful for correctness parity, not day-to-day iteration:

```bash
LLM_MODEL=hermes3 ./download_models.sh                            # ~14.4 GB
export LLM_MODEL_DIR=./models/llm/Hermes-3-Llama-3.2-3B
```

Each model lands in its **own** directory. Don't point `LLM_MODEL_DIR` at a dir
holding two models: the gateway picks the first matching `*.onnx` and would pair
one model's graph with the other's `config.json` — wrong KV shape or wrong stop
token, no error message.

| `LLM_MODEL=` | Model | Size | Notes |
|--------------|-------|------|-------|
| `qwen25` (default) | Qwen2.5-1.5B-Instruct | 1.5 GB int8 | Self-contained, loads in seconds, ~10-20 tok/s; ChatML |
| `hermes3` | Hermes-3-Llama-3.2-3B | 14.4 GB FP32 | Same family as prod; correct for model-family testing |
| `llama32-1b` | Llama-3.2-1B-Instruct | ~1 GB int8 | Kept for comparison; garbled greedy output (verified) |

Then confirm the startup log again — `chat_template` and `eos_ids` are
per-model, and they are what determine whether replies come out sane.

### Where models live

`./download_models.sh` writes everything under **one** root, split per stage:

```
models/
├── stt/    parakeet_{nemo128,encoder.int8,decoder_joint.int8}.onnx + vocab.txt
├── stt_eou/  Parakeet EOU 120M streaming: encoder.onnx (fp16) + decoder_joint.onnx
│           + vocab.txt + config.json + encoder_meta.json
├── sherpa_zipformer/  Sherpa Zipformer: encoder.onnx (fp32, 155 MB)
│           + decoder.onnx + joiner.onnx + tokens.txt
├── tts/    kokoro_v1.0_quantized.onnx, kokoro_tokenizer.json, kokoro_voices/
├── llm/    ONE DIRECTORY PER MODEL:
│           ├── Hermes-3-Llama-3.2-3B/   (model.onnx + model.onnx_data + 4 JSONs)
│           └── Qwen2.5-1.5B-Instruct/
├── vad/    silero_vad.onnx
└── dfn3/   dfn3_{enc,erb_dec,df_dec}.onnx + dfn3_config.ini
```

`models/{stt,tts,llm}` are the gateway's **built-in defaults**, so running from
the repo root needs no path env vars at all. The LLM is the exception: point
`LLM_MODEL_DIR` at a *specific checkpoint dir*, since each has its own
`config.json`. (If `models/llm/` happens to contain exactly one model dir, the
gateway descends into it; with two or more it warns and refuses to guess.)

Override the root with `MODELS_ROOT=/somewhere ./download_models.sh` — then set
`STT_MODEL_DIR` / `TTS_MODEL_DIR` / `LLM_MODEL_DIR` to match. All of `models/`
is gitignored.

## Quick Start (Docker)

```bash
# 1. Download models on the HOST first (the compose file expects them present)
./download_models.sh

# 2. Start the dev stack (gateway only; no GPU needed)
docker compose -f docker-compose.dev.yml up --build gateway

# 3. Verify
curl http://localhost:3000/health
docker compose -f docker-compose.dev.yml logs gateway | grep -i 'chat_template\|pools'
```

The gateway container reads models from the `models` volume at `/app/models`,
mirroring the host's `./models/` layout — so the Hermes dir must land at
`/app/models/llm/Hermes-3-Llama-3.2-3B/`, which is what `LLM_MODEL_DIR` in
[docker-compose.dev.yml](docker-compose.dev.yml) points at. `./download_models.sh`
writes to host paths, so either populate the volume from the host or bind-mount
`./models` instead of using the named volume.

Give the container ≥18 GB of memory (Docker Desktop → Resources): 14.4 GB of
weights plus the KV cache. Under that limit the LLM pool fails to build and the
gateway silently falls back to the stub.

## Architecture (Dev)

```
┌────────────────────────────────────────────────────────────┐
│  Rust Media Gateway (cargo-watch hot-reload)                │
│                                                             │
│  PCM in ─▶ VAD ─▶ STT pool ─▶ LLM pool ─▶ TTS pool ─▶ PCM   │
│           (Silero    (Parakeet)  (Hermes-3   (Kokoro)       │
│            or RMS)                ONNX)                     │
│                                                             │
│  Each pool = bounded queue + N replica threads, one model    │
│  instance per thread, shared by every session.               │
│  <STAGE>_REPLICAS=0 disables that stage; it never loads.     │
│                                                             │
│  WS:     /v1/realtime?dialect=openai|gemini                  │
│  HTTP:   /v1/stt/stream, /v1/llm/stream, /v1/tts/stream      │
│  Health: /health, /v1/pools                                  │
└────────────────────────────────────────────────────────────┘
         │
         │ optional: LLM_BACKEND=vllm   (what prod always does)
         ▼
┌────────────────────────────────────────┐
│  External OpenAI-compatible server      │
│  vLLM / TGI / dev_server.py             │
│  Gateway becomes a streaming SSE client │
└────────────────────────────────────────┘
```

With `LLM_BACKEND=onnx` (the dev default) the gateway loads the LLM weights
itself and no second process exists. Switching to `vllm` changes only where
tokens come from — STT, TTS and VAD stay in-process in both profiles.

## Configuration

Two committed templates hold a complete, working profile each — start from one
rather than assembling vars by hand:

| File | Profile |
|------|---------|
| [env.dev.example](env.dev.example) | ONNX/CPU, single user, no auth, RMS VAD |
| [env.prod.example](env.prod.example) | vLLM/GPU, 300+ sessions, auth + Silero required |

```bash
cp env.dev.example .env.dev && set -a && . ./.env.dev && set +a
```

`.env*` is gitignored; only the `.example` files are committed. Secrets
(`SZCA_API_KEY`, `LLM_API_KEY`) belong in a secret manager, never in the file.

### Pool Sizing

| Env Var | Default | Description |
|---------|---------|-------------|
| `STT_REPLICAS` | 1 | Parakeet STT worker count (0=disabled) |
| `LLM_REPLICAS` | 1 | LLM worker count (0=disabled) |
| `TTS_REPLICAS` | 1 | Kokoro TTS worker count (0=disabled) |

Each replica is one model instance on its own OS thread. With `LLM_BACKEND=onnx`
that means one full copy of the weights — `LLM_REPLICAS=2` on the FP32 dev LLM
wants ~29 GB of RAM. With `LLM_BACKEND=vllm` a replica is just an HTTP/SSE
client, so the number is an in-flight-request budget instead. Capacity is
fungible: `0` disables a stage and its model never loads.

### LLM Backend

| Env Var | Default | Applies to | Description |
|---------|---------|-----------|-------------|
| `LLM_BACKEND` | `onnx` | both | `onnx` (in-process ORT), `vllm`/`tgi` (external) |
| `LLM_MODEL_DIR` | `./models/llm` | onnx | Directory holding `model.onnx` + tokenizer + configs |
| `LLM_ONNX_FILE` | first `*.onnx` | onnx | Override the graph filename inside the dir |
| `LLM_TOKENIZER_FILE` | first `*tokenizer*.json` | onnx | Override the tokenizer filename |
| `LLM_CHAT_TEMPLATE` | auto-detected | onnx | Force `chatml` or `llama` (see below) |
| `LLM_MAX_NEW_TOKENS` | 256 | onnx | Generation cap per turn |
| `LLM_BASE_URL` | `http://localhost:8000` | vllm | vLLM/TGI endpoint (point at the LB, not one worker) |
| `LLM_MODEL` | `hermes-llama-3-8b` | vllm | `--served-model-name` sent in the API request |
| `LLM_API_KEY` | (none) | vllm | Bearer token |
| `LLM_MAX_TOKENS` | 1024 | vllm | Max tokens per generation |
| `LLM_TEMPERATURE` | 0.7 | both | Sampling temperature |

> **`LLM_MODEL` is overloaded.** In [download_models.sh](download_models.sh) it
> selects *which model to download* (`hermes3` | `llama32-1b` | `qwen25`); in the
> gateway it is the *API model name* sent to vLLM and is ignored when
> `LLM_BACKEND=onnx`. Don't export one value expecting it to serve both.

**Chat template detection.** The gateway reads the model's own Jinja
`chat_template` out of `tokenizer_config.json` and probes which special tokens it
emits. It deliberately does *not* infer the format from `config.json`'s
`model_type`: Hermes-3-Llama-3.2-3B reports `llama` but was fine-tuned on ChatML.
Set `LLM_CHAT_TEMPLATE` only to override a wrong detection.

### Audio / VAD

| Env Var | Default | Description |
|---------|---------|-------------|
| `SILERO_VAD_MODEL` | auto-detect | Path to Silero VAD ONNX; empty ⇒ RMS-energy fallback |
| `STT_MODEL_DIR` | `./models/stt` | Parakeet encoder/decoder/vocab dir |
| `TTS_MODEL_DIR` | `./models/tts` | Kokoro model + voices dir |
| `TTS_VOICE` | `af_heart` | Default voice pack |

The RMS fallback is fine for dev and for tests (it's deterministic), but it
mistakes background noise for speech — which shows up as the assistant cutting
itself off. Use real Silero anywhere with a live microphone.

### Server

| Env Var | Default | Description |
|---------|---------|-------------|
| `SZCA_LISTEN_ADDR` | `0.0.0.0` | Bind address |
| `SZCA_PORT` | `3000` | Listen port |
| `SZCA_MAX_SESSIONS` | 1000 | Admission cap, enforced **before** the WS upgrade (excess ⇒ 503) |
| `ORT_DYLIB_PATH` | (none) | `libonnxruntime.{so,dylib}` — the `ort` crate is `load-dynamic` |
| `RUST_LOG` | (none) | Log level: `info`, `debug`, `trace` |
| `SZCA_API_KEY` | (disabled) | Bearer token for `/v1/*` + `/metrics` |

Junk values (`SZCA_PORT=not-a-port`, `SZCA_MAX_SESSIONS=0`) log a warning and
fall back to the default rather than failing the boot.

## Running Tests

```bash
# Everything that needs no weights and no network: unit + e2e integration.
cargo test --manifest-path szca_media_gateway/Cargo.toml

# Just the end-to-end turn contract (PCM -> STT -> LLM -> TTS -> PCM).
cargo test --manifest-path szca_media_gateway/Cargo.toml --test e2e_pipeline

# Full build check, tests and benches included (must be warning-free).
cargo check --manifest-path szca_media_gateway/Cargo.toml --all-targets
```

[tests/e2e_pipeline.rs](szca_media_gateway/tests/e2e_pipeline.rs) drives the real
production turn function (`rt_session::run_turn` — the same one the WebSocket
worker calls) with the deterministic stub stages, so it asserts the actual event
contract rather than a reimplementation of it: event ordering, LLM‖TTS
interleaving (first `audio_delta` before `text_done`), PCM16 sample alignment,
deltas reconstructing `text_done`, barge-in before and mid-generation, both wire
dialects encoding valid JSON, and the VAD utterance boundary.

### Real-weights tests (opt-in)

These load actual models, so they **skip green** when the env var is unset — CI
stays weight-free while a laptop can verify the real thing:

```bash
# Real LLM: loads Hermes-3, streams tokens, checks it stops on its own EOS
# (not the token cap) and honours barge-in. ~2.5 min, mostly weight loading.
set -a && . ./.env.dev && set +a
cargo test --release --manifest-path szca_media_gateway/Cargo.toml \
  --test llm_real_inference -- --nocapture

# Real Silero VAD / DeepFilterNet3
SILERO_VAD_MODEL=./models/vad/silero_vad.onnx DFN3_MODEL_DIR=./models/dfn3 \
cargo test --release --manifest-path szca_media_gateway/Cargo.toml \
  --test silero_real_inference --test dfn3_real_inference -- --nocapture

# Real streaming STT - Parakeet EOU: must decode "hello world" three ways
# (direct, pooled, and fed 20 ms at a time) and beat realtime. ~7 s.
STT_EOU_MODEL_DIR=./models/stt_eou \
cargo test --release --manifest-path szca_media_gateway/Cargo.toml \
  --test stt_eou_real_inference -- --nocapture

# Real streaming STT - Sherpa Zipformer: decodes "Hello World", ~7 s.
SHERPA_MODEL_DIR=./models/sherpa_zipformer \
cargo test --release --manifest-path szca_media_gateway/Cargo.toml \
  --test stt_zipformer_real_inference -- --nocapture

# Both streaming models at once:
SHERPA_MODEL_DIR=./models/sherpa_zipformer STT_EOU_MODEL_DIR=./models/stt_eou \
cargo test --release --manifest-path szca_media_gateway/Cargo.toml \
  --test stt_eou_real_inference --test stt_zipformer_real_inference -- --nocapture
```

These paths are **repo-root-relative** even though cargo runs integration tests
with the CWD set to `szca_media_gateway/` — each test resolves a relative path
against the repo root so the values from `.env.dev` work unchanged. Watch the
output rather than the exit code: an unset var is a deliberate green **skip**, so
`ok` alone does not mean the weights were exercised. A path that is set but
missing says so explicitly.

Expected `llm_real_inference` output — the middle line is the point, a 1-token
reply proves generation stopped at `<|im_end|>` rather than running to the cap:

```
reply: "The capital of France is Paris." (7 deltas)
reply: "OK" (1 tokens)
cancelled after 1 token(s): "Sure"
```

Expected `stt_eou_real_inference` output. These assert on the DECODED TEXT on
purpose: every bug found while building this stage produced silence or plausible
garbage rather than an error, so "inference ran" proves nothing.

```
chunked == single push: "hello world"
EOU streaming: 1.60s audio in 0.105s => 15.2x realtime ("hello world")
pool path: "hello world" partials=["hello world"]
silence -> ""

Expected `stt_zipformer_real_inference` output:

```
Zipformer: 1.60s audio in 0.078s => 20.4x realtime ("Hello World")
pool path: "Hello World" partials=["Hello World"]
20ms-frame feed: "Hello World"
silence -> ""
```
```

### Trying streaming STT

Streaming STT is opt-in behind `STT_BACKEND`. Pass `--with-streaming` to `./download_models.sh` first to fetch the streaming models (~400 MB):

```bash
./download_models.sh --with-streaming
```

Run with `STT_BACKEND=streaming` for Parakeet EOU 120M:

```bash
STT_BACKEND=streaming STT_EOU_MODEL_DIR=./models/stt_eou \
cargo run --release --manifest-path szca_media_gateway/Cargo.toml
```

Or run with `STT_BACKEND=zipformer` for Sherpa Zipformer:

```bash
STT_BACKEND=zipformer SHERPA_MODEL_DIR=./models/sherpa_zipformer \
cargo run --release --manifest-path szca_media_gateway/Cargo.toml
```

Anything other than `streaming` (or `eou`) — including a typo — loads the default
Parakeet. That is deliberate: a typo silently loading the 120M model in place of
the 0.6B one would look like an accuracy regression with no config error to point
at. `curl localhost:3000/health` reports which backend is live.

Use `--release`: a debug-build FP32 3B forward pass is slow enough to look hung.

## API Endpoints

### WebSocket — Realtime Voice

```
ws://localhost:3000/v1/realtime?dialect=openai
```

Bidirectional streaming: audio in → STT → LLM → TTS → audio out. Supports
OpenAI Realtime and Gemini Live wire dialects.

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

# Pool health (replicas, queue_depth, latency: p50/p90/p95/p99 per stage)
curl http://localhost:3000/v1/pools

# Health check
curl http://localhost:3000/health
```

All three are **Server-Sent Events**: incremental events as work is produced,
then one terminal event. Use `curl -N` to see them arrive live rather than
buffered.

| Endpoint | Incremental events | Terminal event |
|----------|-------------------|----------------|
| `/v1/stt/stream` | `partial` — interim transcripts (suppress with `"interim_results": false`) | `final` — `{text, confidence, timestamp}` |
| `/v1/llm/stream` | `token` — one decoded text piece each, as `{text, token_id, logprob, index}` | `eos` — `{text, total_tokens, finish_reason}` |
| `/v1/tts/stream` | `audio_chunk` — `{pcm, sample_rate, duration_ms, sequence}`; `pcm` is base64 PCM16 mono | `eos` — `{total_chunks, total_duration_ms, sample_rate}` |

Any of the three can instead emit a single `error` event — `{"error": "..."}` —
when the pool's queue is full, the pool is shut down, or the inference task
panics. The HTTP status is already `200` by then (SSE headers go out before the
job runs), so a client must treat `error` as a terminal event rather than trusting
the status code.

`finish_reason` is `"stop"` when the model emitted its own EOS token and
`"length"` when the reply was cut off by the request's `max_tokens`. A client
that disconnects mid-stream cancels the job, so the replica stops generating
instead of finishing work nobody will read.

`token_id` and `logprob` on `token` events are reported as `-1` / `0.0`: the pool
surfaces decoded text pieces, not per-token ids or probabilities. They are
placeholders for a future contract, not real values — don't key logic off them.

## Dev vs Prod

| | Dev | Prod |
|--|-----|------|
| LLM runtime | in-process ONNX Runtime (CPU) | vLLM (GPU, A100/L40S) |
| LLM weights | Hermes-3-Llama-3.2-3B, **FP32 ONNX, ~14.4 GB** | Hermes-3-Llama-3.1-8B, FP8 |
| Who generates tokens | the gateway itself | vLLM; the gateway is an SSE client |
| STT / TTS / VAD | in-process ONNX | in-process ONNX (unchanged) |
| Replicas | `STT=1 LLM=1 TTS=1` | `STT=8 LLM=64 TTS=8` |
| Concurrency | 1 (single-user) | 300+ (continuous batching) |
| Latency | seconds per turn | sub-second time-to-first-audio |
| VAD | RMS fallback OK | Silero required |
| Hot reload | cargo-watch | static binary |
| Auth | disabled | `SZCA_API_KEY` required |
| TLS | none | terminated upstream |

Same binary, same wire contract — only `LLM_BACKEND` and the pool sizes change,
so no client code changes between dev and prod.

**Dev is a correctness harness, not a performance preview.** A 3B FP32 model on
CPU generates single-digit tokens/second, which is why
[env.dev.example](env.dev.example) caps `LLM_MAX_NEW_TOKENS=96`. If a turn feels
slow in dev, that is the FP32 CPU forward pass, not the pipeline. Dev deliberately
runs the *same model family* as prod so that what you debug locally is
application behaviour rather than cross-model behaviour differences.

## Further Reading

- [PROJECT.md](PROJECT.md) — architecture, deployment profiles, decisions, backlog
- [README.md](README.md) — what the engine is and the wire dialects it speaks
- [szca_load_test/README.md](szca_load_test/README.md) — WebSocket load generator
- [ARCHIVE_SCALING_PLAN.md](ARCHIVE_SCALING_PLAN.md) — superseded phase plan, kept for history
