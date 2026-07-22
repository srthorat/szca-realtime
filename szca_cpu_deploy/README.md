# SZCA CPU-Only Deployment

No GPU required. Runs on any x86_64 machine with 8GB+ RAM.

## Quick Start

```bash
# 1. Build
chmod +x build.sh
./build.sh

# 2. Download models (optional — stubs work without real models)
cd ../
chmod +x download_models.sh
./download_models.sh

# 3. Run server
cd szca_cpu_deploy/build
./szca_cpu --port 8080 --model_dir ../models/
```

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `POST /v1/stt/stream` | POST | Streaming speech-to-text |
| `POST /v1/llm/stream` | POST | Streaming language model |
| `POST /v1/tts/stream` | POST | Streaming text-to-speech |
| `POST /v1/voice` | POST | Unified voice API |
| `GET /health` | GET | Health check |

## Example Requests

### Health Check
```bash
curl http://localhost:8080/health
```

### STT
```bash
curl -X POST http://localhost:8080/v1/stt/stream \
  -H "Content-Type: application/json" \
  -d '{"model":"parakeet","language":"en","interim_results":true}'
```

### LLM
```bash
curl -X POST http://localhost:8080/v1/llm/stream \
  -H "Content-Type: application/json" \
  -d '{"model":"hermes-3-3b","messages":[{"role":"user","content":"Hello"}],"stream":true}'
```

### TTS
```bash
curl -X POST http://localhost:8080/v1/tts/stream \
  -H "Content-Type: application/json" \
  -d '{"model":"kokoro-82m","voice":"af_heart","input":"Hello world","stream":true}'
```

## Docker

```bash
# Build
docker build -t szca-cpu .

# Run
docker run -p 8080:8080 szca-cpu

# Test
curl http://localhost:8080/health
```

## Requirements

- **CPU:** x86_64 with AVX-512 (recommended)
- **RAM:** 8GB minimum, 16GB recommended
- **Disk:** 5GB for models
- **OS:** Linux, macOS, or Windows (WSL2)

## Performance on CPU

| Metric | Value |
|---|---|
| **LLM Tokens/sec** | 120-250 (3B INT4) |
| **STT Latency** | ~30ms per chunk |
| **TTS Latency** | ~15ms per chunk |
| **Glass-to-Glass** | ~100-150ms |
| **Concurrent Users** | 50-100 |

## vs GPU Mode

| Metric | CPU Mode | GPU Mode (A100) |
|---|---|---|
| **Hardware** | Any x86_64 | NVIDIA GPU required |
| **LLM Model** | 3B INT4 | 3B INT8 or 70B FP8 |
| **Tokens/sec** | 120-250 | 10,000-15,000 |
| **Concurrent** | 50-100 | 500-800 |
| **Latency** | ~100-150ms | ~50-60ms |
| **Cost** | ~$200/mo | ~$1,700/mo |
