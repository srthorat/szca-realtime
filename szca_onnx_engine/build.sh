#!/bin/bash
# SZCA ONNX Engine — Build & Test Script
# Usage: ./build.sh

set -e

echo "=== SZCA ONNX Engine v5.0.0 ==="
echo ""

# Create build directory
mkdir -p build
cd build

# Configure
echo "[1/3] Configuring..."
cmake .. -DCMAKE_BUILD_TYPE=Debug

# Build
echo "[2/3] Building..."
make -j$(sysctl -n hw.ncpu)

# Run tests
echo "[3/3] Running tests..."
./szca_tests

echo ""
echo "=== All tests passed ==="
