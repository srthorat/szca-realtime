# SZCA Inference Service

Real ONNX inference over HTTP for the SZCA voice pipeline — **no stubs**. Every
endpoint runs genuine model inference and has been validated with real audio and
text (including a TTS↔STT round-trip and a full voice→voice pipeline).

This is the **Python reference implementation**: it favors *correctness* by
reusing each model's exact, verified pre/post-processing (NeMo mel frontend as an
ONNX graph, Kokoro's official Misaki G2P, the model's chat template + KV-cache
loop). It doubles as the **correctness oracle** for a future Rust production port
(see "Scalability" below).

## Endpoints

| Method | Path | Input | Output |
|---|---|---|---|
| POST | `/v1/stt` | multipart `file` (WAV/FLAC/OGG) | `{"text": "..."}` |
| POST | `/v1/llm` | `{"prompt"\|"messages", "max_new_tokens", "temperature", "top_p"}` | `{"reply": "...", "tokens": N}` |
| POST | `/v1/tts` | `{"text", "speed"}` | `audio/wav` (24 kHz) |
| POST | `/v1/pipeline` | multipart `file` (voice→voice) | `audio/wav` + `X-Transcript`/`X-Reply` headers (base64) |
| GET | `/health`, `/ready` | — | liveness / readiness |

## Models

| Stage | Model | Chain |
|---|---|---|
| STT | Parakeet TDT 0.6B v3 (int8) | `nemo128` mel → encoder → decoder_joint (TDT greedy) → SentencePiece vocab |
| LLM | Qwen2.5-1.5B-Instruct (int8) | ChatML template → KV-cache autoregressive decode (geometry/EOS/family auto-detected from `config.json`; swap to Llama-3.2-1B/Hermes-3 via env only) |
| TTS | Kokoro-82M v1.0 (quantized) | **Misaki G2P** → phoneme IDs → Kokoro ONNX + `af_heart` style |

Models (~1.9 GB) are **not** baked into the image — mount them read-only. Populate
with `../download_models.sh` first, which writes ONE root split per stage:
`models/{stt,tts,llm,vad,dfn3}`, with each LLM in its own subdirectory under
`llm/`. The Parakeet `nemo128.onnx` mel frontend must be present in the STT dir as
`parakeet_nemo128.onnx`.

## Run

```bash
# From this directory, with the repo-root ../models/ populated:
docker compose up --build
./smoke_test.sh                 # asserts all 4 endpoints on real data
```

Or locally without Docker:

```bash
python3.11 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
ENGINE_MODELS_DIR=../models \
LLM_MODELS_DIR=../models/llm/Qwen2.5-1.5B-Instruct \
uvicorn app.main:app --port 8900 --app-dir .
```

## Configuration (env)

| Var | Default | Meaning |
|---|---|---|
| `ENGINE_MODELS_DIR` | `/models/engine` | Model **root**; stages live in subdirs |
| `STT_MODELS_DIR` | `$ENGINE_MODELS_DIR/stt` | Parakeet model dir |
| `TTS_MODELS_DIR` | `$ENGINE_MODELS_DIR/tts` | Kokoro model dir + `kokoro_voices/` |
| `LLM_MODELS_DIR` | `$ENGINE_MODELS_DIR/llm` | LLM model dir — set this to a **specific checkpoint dir** (e.g. `.../llm/Qwen2.5-1.5B-Instruct`); the `llm/` parent holds one dir per model |
| `TTS_VOICE` | `af_heart` | Kokoro voice pack |
| `ORT_INTRA_OP_THREADS` | `0` (auto) | ORT threads per session |
| `MAX_CONCURRENCY` | CPU count | simultaneous heavy inferences before 503 |
| `ACQUIRE_TIMEOUT_S` | `5.0` | wait for a slot before 503 backpressure |

## Scalability

- Sessions load **once** at startup and are shared; `session.run` is thread-safe
  and releases the GIL, so inference parallelizes in ORT's native thread pool.
- A bounded semaphore caps concurrent heavy inferences and returns **503** when
  saturated (backpressure) rather than collapsing tail latency.
- Scale **horizontally** (`docker compose up --scale inference=N` behind a load
  balancer); each replica loads its own sessions.

**Known ceiling:** the LLM autoregressive loop makes many tiny `run()` calls with
Python glue between each, so the GIL serializes that glue under high concurrency.
For high-concurrency production, port the (here-verified) pre/post-processing into
the Rust `tokio`/`ort` gateway — no GIL, native HuggingFace `tokenizers` — and
validate its output against this service.
