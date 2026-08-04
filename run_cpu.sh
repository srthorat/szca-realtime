#!/bin/bash
# SZCA Media Gateway — Local CPU Sanity & Run Script
#
# Runs the full real-time gateway pipeline on CPU without requiring an NVIDIA GPU.
# Configures optimal local CPU thread pool sizes and ONNX Runtime execution settings.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== SZCA Media Gateway (Local CPU Mode) ==="

# 1. Check for required model weights
if [ ! -f "./models/vad/silero_vad.onnx" ] || [ ! -d "./models/dfn3" ]; then
    echo "⚠️  Model files missing in ./models/. Running download_models.sh..."
    ./download_models.sh
fi

# Select default dev LLM if not specified
if [ -z "${LLM_MODEL_DIR:-}" ]; then
    if [ -d "./models/llm/Qwen2.5-1.5B-Instruct" ]; then
        export LLM_MODEL_DIR="./models/llm/Qwen2.5-1.5B-Instruct"
    elif [ -d "./models/llm/Hermes-3-Llama-3.2-3B" ]; then
        export LLM_MODEL_DIR="./models/llm/Hermes-3-Llama-3.2-3B"
    fi
fi

# Environment configuration for CPU execution
export RUST_LOG="${RUST_LOG:-info,szca_media_gateway=debug}"
export SZCA_WORKER_THREADS="${SZCA_WORKER_THREADS:-4}"
export SZCA_BLOCKING_THREADS="${SZCA_BLOCKING_THREADS:-64}"

# Single worker per stage pool for CPU
export STT_REPLICAS="${STT_REPLICAS:-1}"
export LLM_REPLICAS="${LLM_REPLICAS:-1}"
export TTS_REPLICAS="${TTS_REPLICAS:-1}"

# Enable DeepFilterNet3 DSP noise cancellation
export DFN3_MODEL_DIR="${DFN3_MODEL_DIR:-./models/dfn3}"

echo "Configuration:"
echo "  LLM_MODEL_DIR:      ${LLM_MODEL_DIR:-[Not Loaded / Stub Mode]}"
echo "  STT_REPLICAS:       $STT_REPLICAS"
echo "  LLM_REPLICAS:       $LLM_REPLICAS"
echo "  TTS_REPLICAS:       $TTS_REPLICAS"
echo "  SZCA_WORKER_THREADS: $SZCA_WORKER_THREADS"
echo ""

echo "Starting szca_media_gateway on http://127.0.0.1:8080..."
echo "  - WebSocket endpoint: ws://127.0.0.1:8080/v1/realtime"
echo "  - Health check:       http://127.0.0.1:8080/health"
echo "  - Prometheus metrics: http://127.0.0.1:8080/metrics"
echo "  - Pool stats:         http://127.0.0.1:8080/v1/pools"
echo ""

exec cargo run --manifest-path szca_media_gateway/Cargo.toml --release
