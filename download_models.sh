#!/bin/bash
# SZCA Model Download Script
# Downloads all required ONNX models for the SZCA voice engine.
#
# SECURITY / SUPPLY-CHAIN NOTES (H13):
#   * All downloads use `curl --fail --location --proto '=https' --tlsv1.2`
#     so that HTTP errors fail loudly (instead of writing an HTML error body
#     into a .onnx file) and only TLS 1.2+ HTTPS transport is allowed.
#   * Integrity is verified with SHA-256. Expected hashes are read from
#     environment variables named MODEL_SHA256_<KEY> (see the map below).
#       - If a hash is provided and does NOT match, the script FAILS LOUDLY.
#       - If a hash is NOT provided, a prominent WARNING is printed and the
#         download is left UNVERIFIED (it is never silently trusted).
#   * To pin a model, export its hash before running, e.g.:
#         export MODEL_SHA256_SILERO_VAD="<64-hex-sha256>"
#         export MODEL_SHA256_DEEPFILTERNET3="<64-hex-sha256>"
#         ./download_models.sh
#   * Prefer pinning HuggingFace / GitHub refs to an immutable commit SHA or
#     release tag rather than mutable `master`/`main`. See SILERO_VAD_REF.

set -euo pipefail

MODELS_DIR="./szca_media_gateway/models"
ENGINE_MODELS_DIR="./szca_onnx_engine/models"

mkdir -p "$MODELS_DIR" "$ENGINE_MODELS_DIR"

# ----------------------------------------------------------------------------
# Pinned refs (avoid mutable branches like master/main).
# TODO: replace with the exact commit SHA / release tag you have audited.
# Silero publishes the VAD model on the `master` branch which is MUTABLE;
# pin to a commit or tag to make downloads reproducible & tamper-evident.
SILERO_VAD_REF="master"   # TODO: pin to a commit SHA or tag, e.g. "v5.1"
# ----------------------------------------------------------------------------

echo "=== SZCA Model Download ==="
echo ""

# Detect an available SHA-256 checksum tool (Linux: sha256sum, macOS: shasum).
if command -v sha256sum >/dev/null 2>&1; then
  sha256_of() { sha256sum "$1" | awk '{print $1}'; }
elif command -v shasum >/dev/null 2>&1; then
  sha256_of() { shasum -a 256 "$1" | awk '{print $1}'; }
else
  echo "ERROR: neither 'sha256sum' nor 'shasum' is available; cannot verify integrity." >&2
  exit 1
fi

# download_and_verify <url> <output_path> <sha_env_key>
#   <sha_env_key> selects the env var MODEL_SHA256_<sha_env_key> holding the
#   expected SHA-256. Fails loudly on HTTP error, empty file, or hash mismatch.
download_and_verify() {
  local url="$1"
  local out="$2"
  local key="$3"
  local env_var="MODEL_SHA256_${key}"
  local expected="${!env_var:-}"

  curl --fail --location --proto '=https' --tlsv1.2 -o "$out" "$url"

  # Guard against a "successful" download that produced an empty file.
  if [ ! -s "$out" ]; then
    echo "ERROR: downloaded file is empty: $out (url: $url)" >&2
    rm -f "$out"
    exit 1
  fi

  local actual
  actual="$(sha256_of "$out")"

  if [ -n "$expected" ]; then
    if [ "$actual" != "$expected" ]; then
      echo "ERROR: checksum MISMATCH for $out" >&2
      echo "  expected ($env_var): $expected" >&2
      echo "  actual:              $actual" >&2
      rm -f "$out"
      exit 1
    fi
    echo "  ✓ checksum verified ($env_var)"
  else
    echo "  ⚠️  WARNING: $env_var is not set — integrity of $out is UNVERIFIED."
    echo "     Observed SHA-256: $actual"
    echo "     Pin it with: export $env_var=\"$actual\""
  fi
}

# 1. Silero VAD v5
echo "[1/5] Downloading Silero VAD v5..."
download_and_verify \
  "https://github.com/snakers4/silero-vad/raw/${SILERO_VAD_REF}/src/silero_vad/data/silero_vad.onnx" \
  "$MODELS_DIR/silero_vad.onnx" \
  "SILERO_VAD"
echo "  ✓ Silero VAD downloaded"

# 2. DeepFilterNet3
echo "[2/5] Downloading DeepFilterNet3..."
download_and_verify \
  "https://github.com/Rikorose/DeepFilterNet/releases/download/v0.4.1/deepfilternet3.onnx" \
  "$MODELS_DIR/deepfilternet3.onnx" \
  "DEEPFILTERNET3"
echo "  ✓ DeepFilterNet3 downloaded"

# 3. Parakeet TDT 0.6B V3 (FP16 ONNX)
echo "[3/5] Downloading Parakeet TDT 0.6B V3..."
download_and_verify \
  "https://huggingface.co/thoratsr7/parakeet-tdt-0.6b-v3-onnx/resolve/main/model.onnx" \
  "$ENGINE_MODELS_DIR/parakeet_tdt_0.6b_v3_fp16.onnx" \
  "PARAKEET_TDT"
echo "  ✓ Parakeet TDT downloaded"

# 4. Hermes-3-Llama-3.2-3B (INT8 ONNX)
echo "[4/5] Downloading Hermes-3-Llama-3.2-3B INT8..."
download_and_verify \
  "https://huggingface.co/NousResearch/Hermes-3-Llama-3.2-3B/resolve/main/model.onnx" \
  "$ENGINE_MODELS_DIR/hermes-3-3b-int8.onnx" \
  "HERMES3_3B"
echo "  ✓ Hermes-3 downloaded"

# 5. Kokoro-82M TTS
echo "[5/5] Downloading Kokoro-82M TTS..."
download_and_verify \
  "https://huggingface.co/hexgrad/Kokoro-82M/resolve/main/kokoro-v1.0.onnx" \
  "$ENGINE_MODELS_DIR/kokoro_v1.0.onnx" \
  "KOKORO_MODEL"
download_and_verify \
  "https://huggingface.co/hexgrad/Kokoro-82M/resolve/main/voices.bin" \
  "$ENGINE_MODELS_DIR/voices.bin" \
  "KOKORO_VOICES"
echo "  ✓ Kokoro-82M downloaded"

echo ""
echo "=== Download Complete ==="
echo ""
echo "Models installed to:"
echo "  Gateway: $MODELS_DIR/"
echo "  Engine:  $ENGINE_MODELS_DIR/"
echo ""
echo "Total size:"
du -sh "$MODELS_DIR" "$ENGINE_MODELS_DIR"
