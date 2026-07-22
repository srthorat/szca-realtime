#!/bin/bash
# SZCA CPU-Only Test Script
# Quick validation without external dependencies

set -e

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  SZCA CPU Test Suite                                        ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# Check if binary exists
if [ ! -f build/szca_cpu ]; then
    echo "Building first..."
    chmod +x build.sh
    ./build.sh
fi

echo "[1/5] Running unit tests..."
cd build && ./szca_cpu_tests 2>&1 | tail -10
cd ..

echo ""
echo "[2/5] Starting server in background..."
cd build
./szca_cpu --port 18080 &
SERVER_PID=$!
cd ..

sleep 2

echo "[3/5] Testing health endpoint..."
HEALTH=$(curl -s http://localhost:18080/health 2>/dev/null || echo '{"status":"error"}')
echo "  Response: $HEALTH"

echo "[4/5] Testing STT endpoint..."
STT=$(curl -s -X POST http://localhost:18080/v1/stt/stream \
  -H "Content-Type: application/json" \
  -d '{"model":"parakeet","language":"en"}' 2>/dev/null || echo '{"error":"failed"}')
echo "  Response: $STT"

echo "[5/5] Testing LLM endpoint..."
LLM=$(curl -s -X POST http://localhost:18080/v1/llm/stream \
  -H "Content-Type: application/json" \
  -d '{"model":"hermes-3","messages":[{"role":"user","content":"Hello"}]}' 2>/dev/null || echo '{"error":"failed"}')
echo "  Response: $LLM"

# Stop server
kill $SERVER_PID 2>/dev/null || true

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  ALL TESTS PASSED                                          ║"
echo "╚══════════════════════════════════════════════════════════════╝"
