#!/bin/bash
# SZCA CPU-Only Build & Test Script
# No GPU required — runs on any x86_64 machine

set -e

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  SZCA CPU-Only Deployment v5.0.0                            ║"
echo "║  No GPU required — testing on CPU only                      ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# Create build directory
mkdir -p build
cd build

# Configure
echo "[1/4] Configuring CMake (CPU-only)..."
cmake .. -DCMAKE_BUILD_TYPE=Debug \
         -DCMAKE_CXX_FLAGS="-march=native -O2"

# Build
echo "[2/4] Building..."
make -j$(sysctl -n hw.ncpu 2>/dev/null || echo 4)

# Run tests
echo "[3/4] Running tests..."
./szca_cpu_tests

# Run server (optional)
echo ""
echo "[4/4] Starting CPU server..."
echo "  API: http://0.0.0.0:8080/v1/realtime"
echo "  Press Ctrl+C to stop"
echo ""
./szca_cpu --port 8080 --model_dir ../models/
