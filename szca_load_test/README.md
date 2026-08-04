# Load Testing

## Prerequisites

```bash
# Install k6
# macOS
brew install k6

# Ubuntu
sudo apt install -y k6
```

## HTTP API Load Test

Tests `/health`, `/v1/stt/stream`, `/v1/llm/stream`, `/v1/tts/stream`, `/v1/pools`.

```bash
# Start the gateway first
cd ../szca_media_gateway && cargo run

# Run load test
k6 run load_test.js
```

Environment variables:
- `BASE_URL` — target URL (default: `http://localhost:8080`)

## WebSocket Realtime Load Test

Tests `/v1/realtime` with simulated voice sessions (connect → configure → 3 turns → close).

```bash
k6 run ws_load_test.js
```

Environment variables:
- `BASE_URL` — target host:port (default: `localhost:3000`)
- `DIALECT` — wire format: `openai` or `gemini` (default: `openai`)

## Targets (300 sessions on 16c/64GB VM)

| Metric | Target |
|--------|--------|
| WS connections sustained | 100+ |
| Turn latency (p95) | < 2s |
| HTTP error rate | < 1% |
| WS error rate | < 5% |
