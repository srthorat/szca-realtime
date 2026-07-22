#!/bin/bash
# SZCA C++ Engine — Build & Test Script
# Run this manually to verify C++ compilation

set -e

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  SZCA C++ Engine Build & Test                               ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# Navigate to project root
cd "$(dirname "$0")"

# Create build directory
echo "[1/4] Creating build directory..."
mkdir -p build
cd build

# Configure
echo "[2/4] Configuring CMake..."
cmake .. -DCMAKE_BUILD_TYPE=Debug \
         -DCMAKE_CXX_FLAGS="-std=c++20 -Wall -Wextra"

# Build
echo "[3/4] Building..."
make -j$(sysctl -n hw.ncpu 2>/dev/null || echo 4)

# Run tests
echo "[4/4] Running tests..."
if [ -f szca_tests ]; then
    ./szca_tests
    echo ""
    echo "═══════════════════════════════════════════════════════════════"
    echo "  ✅ All C++ tests passed"
    echo "═══════════════════════════════════════════════════════════════"
else
    echo "  ❌ szca_tests not found — build may have failed"
    exit 1
fi
