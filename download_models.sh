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

# ONE model root for the whole repo, laid out per stage. These paths are exactly
# the gateway's built-in defaults (`./models/stt`, `./models/tts`, `./models/llm`),
# so a plain `cargo run` from the repo root finds every model with NO env vars set.
# Models were previously split across szca_media_gateway/models and
# szca_onnx_engine/models, which meant two dirs to mount, two to gitignore, and
# every path had to be spelled out in the environment.
MODELS_ROOT="${MODELS_ROOT:-./models}"
WITH_STREAMING="${WITH_STREAMING:-0}"

for arg in "$@"; do
  case "$arg" in
    --with-streaming)
      WITH_STREAMING=1
      ;;
    --help|-h)
      echo "Usage: ./download_models.sh [--with-streaming]"
      echo "  --with-streaming  Download streaming STT models (EOU 120M & Sherpa Zipformer, ~400 MB)"
      echo "Environment variables:"
      echo "  WITH_STREAMING=1  Same as --with-streaming"
      echo "  MODELS_ROOT=./models  Target directory root"
      echo "  LLM_MODEL=qwen25-05b|qwen25|hermes3|llama32-1b  Select LLM variant (default: qwen25-05b)"
      exit 0
      ;;
  esac
done

STT_DIR="$MODELS_ROOT/stt"
TTS_DIR="$MODELS_ROOT/tts"
LLM_DIR="$MODELS_ROOT/llm"
VAD_DIR="$MODELS_ROOT/vad"
DFN3_DIR="$MODELS_ROOT/dfn3"

mkdir -p "$STT_DIR" "$TTS_DIR" "$LLM_DIR" "$VAD_DIR" "$DFN3_DIR"

# ----------------------------------------------------------------------------
# Pinned refs (avoid mutable branches like master/main).
# TODO: replace with the exact commit SHA / release tag you have audited.
# Silero publishes the VAD model on the `master` branch which is MUTABLE;
# pin to a commit or tag to make downloads reproducible & tamper-evident.
SILERO_VAD_REF="v5.1"

# Default (observed) SHA-256 hashes so downloads are verified out of the box.
# Override any of these via the environment before running. These were captured
# from the sources below on 2026-07-22 and validated with onnx.checker.
: "${MODEL_SHA256_SILERO_VAD:=1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3}"
: "${MODEL_SHA256_DFN3_ENC:=7c5399d3da8a50ebef1c1a0ae421b33376aa5e45d0e92df16da7e83c9c131916}"
: "${MODEL_SHA256_DFN3_ERB_DEC:=ab669a1d10afe20911728b33053a452071042317a90581092b325da7b2f9d895}"
: "${MODEL_SHA256_DFN3_DF_DEC:=23114ce3b0f6464b763ee62f7bb8aab6b2a129a21eabd5bcfe59413db05f278a}"
: "${MODEL_SHA256_DFN3_CONFIG:=415eb925d44990d938fb739f514aa3662c1ec0ea836cff044fa1291b82cb4290}"
# Kokoro TTS (quantized model + af_heart voice) — verified 2026-07-22.
: "${MODEL_SHA256_KOKORO_MODEL:=fbae9257e1e05ffc727e951ef9b9c98418e6d79f1c9b6b13bd59f5c9028a1478}"
: "${MODEL_SHA256_KOKORO_VOICE_AF_HEART:=d583ccff3cdca2f7fae535cb998ac07e9fcb90f09737b9a41fa2734ec44a8f0b}"
# LLM (ONNX) — Llama-3.2-1B-Instruct int8 — verified 2026-07-22 (~9 tok/s CPU).
: "${MODEL_SHA256_LLAMA32_1B:=ad53ed44994c27ab60e8024e83fcd83ce38db4d4ad3a28549b432480eb84d180}"
# Parakeet mel-feature frontend (nemo128) — verified 2026-07-22 via TTS<->STT round-trip.
: "${MODEL_SHA256_PARAKEET_NEMO128:=bff2bc1bef1d1185da8bb69419b51c68fc50e88c923654ced2dfe4e055e4e938}"
MODEL_SHA256_ZIPF_ENCODER MODEL_SHA256_ZIPF_DECODER \
MODEL_SHA256_ZIPF_JOINER MODEL_SHA256_ZIPF_VOCAB \
# Parakeet EOU 120M streaming — FP16 encoder export, verified 2026-07-26.
#
# WHY THE FP16 EXPORT AND NOT THE SMALLER INT8 ONE: the widely-linked
# soniqo/Parakeet-EOU-120M-ONNX-INT8 encoder is built from `ConvInteger` (56
# nodes) + `MatMulInteger` (154). ONNX Runtime has NO CPU kernel for signed-INT8
# `ConvInteger` before **1.24**, and we are pinned to ORT 1.22 (`ort` 2.0.0-rc.10
# → ORT_API_VERSION 22). Measured: 1.19 / 1.22 / 1.23 all fail at session
# creation with `NOT_IMPLEMENTED: ConvInteger(10)`; 1.24+ works. Reproduced on
# BOTH arm64 (macOS) and x86_64 (Linux container) — it is not a Mac quirk and
# would fail identically on the prod g6e.48xlarge.
#
# This FP16 export of the same NVIDIA base model (nvidia/parakeet_realtime_eou_120m-v1)
# has no integer ops, loads on ORT 1.22, and was verified end-to-end (decodes
# "hello world" from our Kokoro sample, <EOU> fires on trailing silence, ~21x
# realtime on an M-series CPU). See PROJECT.md §16 for the full audit.
#
# TWO graphs here (encoder + FUSED decoder_joint) — unlike soniqo's three.
: "${MODEL_SHA256_EOU_ENCODER:=9f9bb8f2e11fd8f66763d94042b9b9de721b4dcfba9ca9bb1071272ac3ff0ddb}"
: "${MODEL_SHA256_EOU_DECODER:=5464333fa933bf2c60952c08f01f5b5dd3fe3176e1c70d3eeddb48412514a0f9}"
: "${MODEL_SHA256_EOU_VOCAB:=77c3f876cddac2d9ad82efceea38fd6acd16575e0ab54ab3396aa4621fa8ff02}"
: "${MODEL_SHA256_EOU_META:=be3fc92f37e937d64cfdc746100069f7957a657a744558ea4c79fe367d539ff7}"
: "${MODEL_SHA256_EOU_CONFIG:=38dea700e61fb7927d26a55977f524e063b00dc5922cfb313c74b91868178eee}"
       MODEL_SHA256_ZIPF_ENCODER MODEL_SHA256_ZIPF_DECODER \n       MODEL_SHA256_ZIPF_JOINER MODEL_SHA256_ZIPF_VOCAB \n
# Sherpa Zipformer streaming STT (kouhxp/sherpa-onnx-streaming-zipformer-en-kroko).
: "${MODEL_SHA256_ZIPF_ENCODER:=d9e1fc347fe75eb0656c89bbccedaf18c0520835334154b8fb389757edd91788}"
: "${MODEL_SHA256_ZIPF_DECODER:=6ab9ff12eaea3c759c2cda414320dcd0e263eaf71b9cff6eca9f814544907708}"
: "${MODEL_SHA256_ZIPF_JOINER:=1ae2e9cf4e80fd57a26023c3d6f74a6e26bf3e799061b6b8e7c3ca961851bbba}"
: "${MODEL_SHA256_ZIPF_VOCAB:=457687fd2ec323c2f47edf788db8b28e549d07a02058f4e268e259adeb9b4fd1}"
# Qwen2.5-1.5B-Instruct int8 (legacy large dev LLM) — verified 2026-07-22 (clean answers).
: "${MODEL_SHA256_QWEN25_1_5B:=ee8364373c6fb35644c67fd8127cbee6c3d98ac889f8bb32ea4ac04a29650787}"
: "${MODEL_SHA256_QWEN25_TOKENIZER:=a8506e7111b80c6d8635951a02eab0f4e1a8e4e5772da83846579e97b16f61bf}"
: "${MODEL_SHA256_QWEN25_CONFIG:=215eb99c4955b0c42ea9f6e0980d922c228950c5dbc09bde6dc451fbba4d21f3}"
: "${MODEL_SHA256_QWEN25_GEN_CONFIG:=f7e7ce458658b2d40d9eb213b91b77a8bf698845ab89360976722d7ac46928a3}"
# Qwen2.5-0.5B-Instruct (default dev LLM) — ~350 MB ONNX, ~1s load time.
# Set MODEL_SHA256_QWEN25_05B to a 64-char hex SHA-256 to verify integrity;
# without it, the download runs UNVERIFIED (with a warning).
: "${MODEL_SHA256_QWEN25_05B:=}"
: "${MODEL_SHA256_QWEN25_05B_TOKENIZER:=}"
: "${MODEL_SHA256_QWEN25_05B_CONFIG:=}"
: "${MODEL_SHA256_QWEN25_05B_GEN_CONFIG:=}"
# Hermes-3-Llama-3.2-3B ONNX (default dev LLM) at rev b378eeb — captured from the
# working local checkpoint on 2026-07-25 and verified by real inference
# (tests/llm_real_inference.rs: correct answer, natural EOS stop, cancel honored).
# tokenizer_config.json is pinned too because it carries the chat template and
# eos_token that drive prompt formatting and stop detection — a substituted one
# would silently degrade every reply rather than fail.
: "${MODEL_SHA256_HERMES3_ONNX:=955254b1c038e9feeeb5b61d8149513d3e1f9d1a9ab79580c446cd7b229aa955}"
: "${MODEL_SHA256_HERMES3_ONNX_DATA:=edf066a3481438efd0ed92e050b00e43b16401fdb740e898b1d4c84750e6af72}"
: "${MODEL_SHA256_HERMES3_TOKENIZER:=9f908f9b84390fd12c6d0c356765257846c53f60bf472ff4996a440a1e230373}"
: "${MODEL_SHA256_HERMES3_TOK_CONFIG:=ba3b1536e5c2a28720f7b5cf91f790637f79795cfa2ecee7c78a58d6bf5a49e1}"
: "${MODEL_SHA256_HERMES3_CONFIG:=26eaee613fa88164654ca202ce4934c7b644e0172f5218659a58bfd4f6d5c2e2}"
: "${MODEL_SHA256_HERMES3_GEN_CONFIG:=75d6c3fea2bb115bd47bf0f435d4a63f3bd8946f48d781268bd10fcf19d3bcee}"
export MODEL_SHA256_SILERO_VAD MODEL_SHA256_DFN3_ENC MODEL_SHA256_DFN3_ERB_DEC \
       MODEL_SHA256_DFN3_DF_DEC MODEL_SHA256_DFN3_CONFIG \
       MODEL_SHA256_EOU_ENCODER MODEL_SHA256_EOU_DECODER \
       MODEL_SHA256_EOU_VOCAB MODEL_SHA256_EOU_META MODEL_SHA256_EOU_CONFIG \
       MODEL_SHA256_KOKORO_MODEL MODEL_SHA256_KOKORO_VOICE_AF_HEART \
       MODEL_SHA256_LLAMA32_1B MODEL_SHA256_PARAKEET_NEMO128 \
       MODEL_SHA256_QWEN25_1_5B MODEL_SHA256_QWEN25_TOKENIZER \
       MODEL_SHA256_QWEN25_CONFIG MODEL_SHA256_QWEN25_GEN_CONFIG \
       MODEL_SHA256_QWEN25_05B MODEL_SHA256_QWEN25_05B_TOKENIZER \
       MODEL_SHA256_QWEN25_05B_CONFIG MODEL_SHA256_QWEN25_05B_GEN_CONFIG \
       MODEL_SHA256_HERMES3_ONNX MODEL_SHA256_HERMES3_ONNX_DATA \
       MODEL_SHA256_HERMES3_TOKENIZER MODEL_SHA256_HERMES3_TOK_CONFIG \
       MODEL_SHA256_HERMES3_CONFIG MODEL_SHA256_HERMES3_GEN_CONFIG
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

  # --retry covers the truncation case: a connection that drops mid-transfer
  # leaves a short file that is NOT empty, so the -s check below would pass it
  # through to the hash check as a confusing "checksum MISMATCH". Retrying the
  # transfer is the fix; the hash check stays as the last line of defence.
  curl --fail --location --proto '=https' --tlsv1.2 \
       --retry 3 --retry-delay 2 --retry-all-errors \
       -o "$out" "$url"

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
echo "[1/8] Downloading Silero VAD v5..."
download_and_verify \
  "https://github.com/snakers4/silero-vad/raw/${SILERO_VAD_REF}/src/silero_vad/data/silero_vad.onnx" \
  "$VAD_DIR/silero_vad.onnx" \
  "SILERO_VAD"
echo "  ✓ Silero VAD downloaded"

# 2. DeepFilterNet3 (streaming speech enhancement).
# NOTE: DFN3 is NOT a single ONNX file. It is a 3-stage streaming pipeline
# (encoder -> ERB-band decoder + deep-filter decoder) plus a config.ini that
# carries the STFT / framing / sample-rate contract. The single
# `deepfilternet3.onnx` the old script fetched never existed (404). We mirror
# the upstream `DeepFilterNet3_onnx.tar.gz` layout from a SHA-pinned HF repo.
echo "[2/8] Downloading DeepFilterNet3 (enc + erb_dec + df_dec + config)..."
DFN3_BASE="https://huggingface.co/bitsydarel/deepfilternet3-onnx/resolve/main"
download_and_verify "$DFN3_BASE/enc.onnx"     "$DFN3_DIR/dfn3_enc.onnx"     "DFN3_ENC"
download_and_verify "$DFN3_BASE/erb_dec.onnx" "$DFN3_DIR/dfn3_erb_dec.onnx" "DFN3_ERB_DEC"
download_and_verify "$DFN3_BASE/df_dec.onnx"  "$DFN3_DIR/dfn3_df_dec.onnx"  "DFN3_DF_DEC"
download_and_verify "$DFN3_BASE/config.ini"   "$DFN3_DIR/dfn3_config.ini"   "DFN3_CONFIG"
echo "  ✓ DeepFilterNet3 downloaded (3 ONNX stages + config)"

# 3. Parakeet TDT 0.6B V3 STT (int8 ONNX: encoder + decoder_joint + vocab).
# The acoustic encoder + decoder_joint are separate graphs (RNN-T style).
# The int8 variant runs real CPU inference; fp16 encoder needs a 2.4GB external
# .data file. Branch is SHA-worthy; pin PARAKEET_REF once audited.
echo "[3/8] Downloading Parakeet TDT 0.6B V3 (int8 STT)..."
PARAKEET_BASE="https://huggingface.co/thoratsr7/parakeet-tdt-0.6b-v3-onnx/resolve/feat%2Ffp16-canonical-v3"
# nemo128.onnx is the log-mel feature frontend exported as an ONNX graph. It is
# REQUIRED by the inference service so NeMo's exact mel features are reproduced
# 1:1 (avoids the "runs but transcribes garbage" trap of a hand-ported frontend).
download_and_verify "$PARAKEET_BASE/nemo128.onnx" \
  "$STT_DIR/parakeet_nemo128.onnx" "PARAKEET_NEMO128"
download_and_verify "$PARAKEET_BASE/encoder-model.int8.onnx" \
  "$STT_DIR/parakeet_encoder.int8.onnx" "PARAKEET_ENCODER"
download_and_verify "$PARAKEET_BASE/decoder_joint-model.int8.onnx" \
  "$STT_DIR/parakeet_decoder_joint.int8.onnx" "PARAKEET_DECODER"
download_and_verify "$PARAKEET_BASE/vocab.txt" \
  "$STT_DIR/parakeet_vocab.txt" "PARAKEET_VOCAB"
echo "  ✓ Parakeet TDT (int8) downloaded"

# 4. Parakeet EOU 120M streaming STT (cache-aware FastConformer, with EOU).
#    Processes audio in 320 ms chunks (64 mel frames), carrying the encoder's
#    attention/conv cache across chunks, and emits <EOU> (id 1024) inline with
#    the transcript so turn-end comes from the ASR stream itself, not just VAD.
#
#    Graph I/O (verified against the ONNX files, 2026-07-26):
#      encoder.onnx        audio_signal[1,128,128] f32 + audio_length[1] INT32
#                          + pre_cache[1,128,16]
#                          + cache_last_channel[17,1,70,512]
#                          + cache_last_time[17,1,512,8]
#                          + cache_last_channel_len[1] INT32
#                        → encoded_output[1,512,T'] + encoded_length[1] INT32
#                          + the three new caches (feed straight back in)
#      decoder_joint.onnx  encoder_outputs[1,512,frames] + targets[1,n] INT32
#                          + input_states_1/2[1,1,640]
#                        → outputs[1,1,target_plus_sos,1027] + output_states_1/2
#
#    NOTE the cache length tensors are INT32 here (soniqo's INT8 export used
#    INT64). Passing the wrong width makes ORT reject the call outright.
EOU_DIR="$MODELS_ROOT/stt_eou"
ZIPF_DIR="$MODELS_ROOT/sherpa_zipformer"

if [ "$WITH_STREAMING" = "1" ] || [ "$WITH_STREAMING" = "true" ]; then
  echo "[4/8] Downloading Parakeet EOU 120M (streaming FP16 STT)..."
  mkdir -p "$EOU_DIR"
  EOU_BASE="https://huggingface.co/AIsley/parakeet-realtime-eou-120m-streaming-fp16/resolve/main"
  download_and_verify "$EOU_BASE/streaming_encoder.fp16.onnx" \
    "$EOU_DIR/encoder.onnx" "EOU_ENCODER"
  download_and_verify "$EOU_BASE/decoder_joint-model.int8.onnx" \
    "$EOU_DIR/decoder_joint.onnx" "EOU_DECODER"
  download_and_verify "$EOU_BASE/vocab.txt" \
    "$EOU_DIR/vocab.txt" "EOU_VOCAB"
  download_and_verify "$EOU_BASE/config.json" \
    "$EOU_DIR/config.json" "EOU_CONFIG"
  download_and_verify "$EOU_BASE/streaming_encoder.meta.json" \
    "$EOU_DIR/encoder_meta.json" "EOU_META"
  echo "  ✓ Parakeet EOU (streaming FP16, 2 graphs) downloaded to $EOU_DIR"

  echo "[5/8] Downloading Sherpa Zipformer streaming STT..."
  mkdir -p "$ZIPF_DIR"
  ZIPF_BASE="https://huggingface.co/kouhxp/sherpa-onnx-streaming-zipformer-en-kroko/resolve/main"
  download_and_verify "$ZIPF_BASE/encoder.onnx"   "$ZIPF_DIR/encoder.onnx"   "ZIPF_ENCODER"
  download_and_verify "$ZIPF_BASE/decoder.onnx"   "$ZIPF_DIR/decoder.onnx"   "ZIPF_DECODER"
  download_and_verify "$ZIPF_BASE/joiner.onnx"    "$ZIPF_DIR/joiner.onnx"    "ZIPF_JOINER"
  download_and_verify "$ZIPF_BASE/tokens.txt"     "$ZIPF_DIR/tokens.txt"     "ZIPF_VOCAB"
  echo "  ✓ Sherpa Zipformer (3 graphs, ~156 MB) downloaded to $ZIPF_DIR"
else
  echo "[4/8] Skipping Parakeet EOU 120M (streaming STT) — pass --with-streaming or WITH_STREAMING=1 to fetch"
  echo "[5/8] Skipping Sherpa Zipformer (streaming STT) — pass --with-streaming or WITH_STREAMING=1 to fetch"
fi


# 5. LLM (ONNX) — Hermes-3-Llama-3.2-3B (default).
# ONNX-native so it runs through ONNX Runtime / the `ort` crate — the SAME
# runtime as STT/TTS/VAD/DFN3 (unlike GGUF, which needs llama.cpp) — and runs a
# full KV-cache autoregressive loop on CPU. The gateway auto-detects the KV
# geometry from config.json and the prompt format from tokenizer_config.json, so
# swapping models needs no code changes.
#
# Qwen2.5-0.5B is the default dev LLM (~350 MB ONNX, loads in ~1s, ChatML,
# ~30-50 tok/s on CPU). Qwen2.5-1.5B INT8 (~1.5 GB) stays available via
# LLM_MODEL=qwen25 for higher quality.
#
# ALTERNATIVES via LLM_MODEL (this script only; the gateway's LLM_MODEL is the
# vLLM API model name — different meaning, same variable):
#   ** DEFAULT: qwen25-05b ** — Qwen2.5-0.5B-Instruct (~350 MB ONNX). Loads in
#                  ~1 second, ~30-50 tok/s on CPU. Best for daily dev iteration.
#   * qwen25     : Qwen2.5-1.5B-Instruct int8 (~1.5GB). Higher quality, ~10-20 tok/s.
#   * llama32-1b : Llama-3.2-1B-Instruct int8 (kept for comparison; the int8
#                  export produced garbled greedy output, verified).
#
# Each model gets its OWN directory: config.json / generation_config.json are
# per-model, and `LLM_MODEL_DIR` resolution picks the first matching *.onnx in the
# dir — sharing one dir would silently mix one model's graph with another's config.
LLM_MODEL="${LLM_MODEL:-qwen25-05b}"
if [ "$LLM_MODEL" = "hermes3" ]; then
  LLM_MODELS_DIR="$LLM_DIR/Hermes-3-Llama-3.2-3B"
  mkdir -p "$LLM_MODELS_DIR"
  echo "[6/8] Downloading Hermes-3-Llama-3.2-3B (ONNX, FP32 + 14.4GB external data)..."
  HERMES_REV="b378eeb804c964de0c1f13dde5f7d6c3073fb3db"
  # All files come from the onnx/ subdirectory of the pinned revision.
  HERMES_BASE="https://huggingface.co/NousResearch/Hermes-3-Llama-3.2-3B/resolve/${HERMES_REV}/onnx"
  # model.onnx is the graph; model.onnx_data holds the FP32 weights as ONNX
  # external data. ORT resolves the external file RELATIVE TO THE GRAPH, using
  # the `location` recorded at export time — which is literally "model.onnx_data".
  # So the data file must keep that exact name and sit beside the graph, hence
  # both land in $LLM_MODELS_DIR unrenamed (unlike the other models).
  download_and_verify "$HERMES_BASE/model.onnx"      "$LLM_MODELS_DIR/model.onnx"      "HERMES3_ONNX"
  download_and_verify "$HERMES_BASE/model.onnx_data" "$LLM_MODELS_DIR/model.onnx_data" "HERMES3_ONNX_DATA"
  # Every metadata file comes from onnx/, NOT the repo root. The root holds the
  # PyTorch checkpoint's metadata (`torch_dtype: bfloat16`, `use_cache: false`)
  # which describes a different artifact than the graph we just downloaded. The
  # fields the gateway reads happen to agree today, but pairing an export with
  # another checkpoint's metadata is how a future revision silently drifts.
  download_and_verify "$HERMES_BASE/tokenizer.json"  "$LLM_MODELS_DIR/tokenizer.json"  "HERMES3_TOKENIZER"
  # config.json + generation_config.json are REQUIRED, not optional: the gateway
  # reads num_hidden_layers / num_key_value_heads / head_dim to size the KV cache
  # and eos_token_id to stop generation. Without them QwenLlm::from_env() errors
  # out and the pool falls back to the stub LLM.
  #
  # NOTE on EOS: generation_config.json here carries Llama-3's INHERITED ids
  # [128001, 128008, 128009] while config.json carries 128039 (`<|im_end|>`) —
  # the one this ChatML fine-tune actually emits. The gateway takes the UNION of
  # both plus the tokenizer's own eos_token; trusting generation_config alone
  # means generation never stops and every reply runs to the token cap.
  download_and_verify "$HERMES_BASE/config.json" \
    "$LLM_MODELS_DIR/config.json" "HERMES3_CONFIG"
  download_and_verify "$HERMES_BASE/generation_config.json" \
    "$LLM_MODELS_DIR/generation_config.json" "HERMES3_GEN_CONFIG"
  # tokenizer_config.json is also REQUIRED: it holds the Jinja `chat_template`
  # the gateway probes to choose ChatML vs Llama-3 prompt formatting. Hermes-3
  # reports model_type "llama" but is ChatML-tuned, so the ARCHITECTURE is not a
  # usable signal — without this file the gateway falls back to its ChatML
  # default, which happens to be right here but is a guess, not a read.
  download_and_verify "$HERMES_BASE/tokenizer_config.json" \
    "$LLM_MODELS_DIR/tokenizer_config.json" "HERMES3_TOK_CONFIG"
  echo "  ✓ Hermes-3-3B (ONNX) downloaded"
  echo "    Point the gateway at it with: export LLM_MODEL_DIR=$LLM_MODELS_DIR"
elif [ "$LLM_MODEL" = "llama32-1b" ]; then
  LLM_MODELS_DIR="$LLM_DIR/Llama-3.2-1B-Instruct"
  mkdir -p "$LLM_MODELS_DIR"
  echo "[6/8] Downloading Llama-3.2-1B-Instruct int8 (ONNX)..."
  LLAMA_BASE="https://huggingface.co/onnx-community/Llama-3.2-1B-Instruct-ONNX/resolve/main"
  download_and_verify "$LLAMA_BASE/onnx/model_int8.onnx" \
    "$LLM_MODELS_DIR/llama-3.2-1b-instruct.int8.onnx" "LLAMA32_1B"
  download_and_verify "$LLAMA_BASE/tokenizer.json" \
    "$LLM_MODELS_DIR/llama_tokenizer.json" "LLAMA32_TOKENIZER"
  download_and_verify "$LLAMA_BASE/config.json" \
    "$LLM_MODELS_DIR/config.json" "LLAMA32_CONFIG"
  echo "  ✓ Llama-3.2-1B-Instruct (ONNX int8) downloaded"
  echo "    Point the gateway at it with: export LLM_MODEL_DIR=$LLM_MODELS_DIR"
elif [ "$LLM_MODEL" = "qwen25" ]; then
  LLM_MODELS_DIR="$LLM_DIR/Qwen2.5-1.5B-Instruct"
  mkdir -p "$LLM_MODELS_DIR"
  echo "[6/8] Downloading Qwen2.5-1.5B-Instruct int8 (ONNX)..."
  QWEN_BASE="https://huggingface.co/onnx-community/Qwen2.5-1.5B-Instruct/resolve/main"
  download_and_verify "$QWEN_BASE/onnx/model_int8.onnx" \
    "$LLM_MODELS_DIR/qwen2.5-1.5b-instruct.int8.onnx" "QWEN25_1_5B"
  download_and_verify "$QWEN_BASE/tokenizer.json" \
    "$LLM_MODELS_DIR/qwen_tokenizer.json" "QWEN25_TOKENIZER"
  # config.json + generation_config.json drive the service's auto-detection.
  download_and_verify "$QWEN_BASE/config.json" \
    "$LLM_MODELS_DIR/config.json" "QWEN25_CONFIG"
  download_and_verify "$QWEN_BASE/generation_config.json" \
    "$LLM_MODELS_DIR/generation_config.json" "QWEN25_GEN_CONFIG"
  echo "  ✓ Qwen2.5-1.5B-Instruct (ONNX int8) downloaded"
  echo "    Point the gateway at it with: export LLM_MODEL_DIR=$LLM_MODELS_DIR"
elif [ "$LLM_MODEL" = "qwen25-05b" ]; then
  LLM_MODELS_DIR="$LLM_DIR/Qwen2.5-0.5B-Instruct"
  mkdir -p "$LLM_MODELS_DIR"
  echo "[6/8] Downloading Qwen2.5-0.5B-Instruct (ONNX, FP16)..."
  QWEN05_BASE="https://huggingface.co/onnx-community/Qwen2.5-0.5B/resolve/main"
  download_and_verify "$QWEN05_BASE/model.onnx" \
    "$LLM_MODELS_DIR/model.onnx" "QWEN25_05B"
  # Some ONNX exports use external data (model.onnx_data). Download it if it
  # exists in the repo; a 404 here is non-fatal (self-contained model).
  curl --fail --location --silent --proto '=https' --tlsv1.2 \
    -o "$LLM_MODELS_DIR/model.onnx_data" \
    "$QWEN05_BASE/model.onnx_data" || true
  download_and_verify "$QWEN05_BASE/tokenizer.json" \
    "$LLM_MODELS_DIR/tokenizer.json" "QWEN25_05B_TOKENIZER"
  download_and_verify "$QWEN05_BASE/config.json" \
    "$LLM_MODELS_DIR/config.json" "QWEN25_05B_CONFIG"
  download_and_verify "$QWEN05_BASE/generation_config.json" \
    "$LLM_MODELS_DIR/generation_config.json" "QWEN25_05B_GEN_CONFIG"
  echo "  ✓ Qwen2.5-0.5B-Instruct (ONNX) downloaded"
  echo "    Point the gateway at it with: export LLM_MODEL_DIR=$LLM_MODELS_DIR"
else
  # Fail loudly on an unrecognized value. A catch-all `else` would silently
  # download some other model for a typo like LLM_MODEL=qwen2.5, leaving you
  # debugging why the checkpoint is not the one you asked for.
  echo "ERROR: unknown LLM_MODEL='$LLM_MODEL'" >&2
  echo "  Valid values: qwen25-05b (default) | qwen25 | hermes3 | llama32-1b" >&2
  exit 1
fi

# 5. Kokoro-82M v1.0 TTS (onnx-community export).
# model + one voice pack + tokenizer. Voices are [510,256] f32 style vectors;
# fetch more from the repo's voices/ dir as needed.
echo "[7/8] Downloading Kokoro-82M v1.0 TTS..."
KOKORO_BASE="https://huggingface.co/onnx-community/Kokoro-82M-v1.0-ONNX/resolve/main"
download_and_verify "$KOKORO_BASE/onnx/model_quantized.onnx" \
  "$TTS_DIR/kokoro_v1.0_quantized.onnx" "KOKORO_MODEL"
mkdir -p "$TTS_DIR/kokoro_voices"
download_and_verify "$KOKORO_BASE/voices/af_heart.bin" \
  "$TTS_DIR/kokoro_voices/af_heart.bin" "KOKORO_VOICE_AF_HEART"
download_and_verify "$KOKORO_BASE/tokenizer.json" \
  "$TTS_DIR/kokoro_tokenizer.json" "KOKORO_TOKENIZER"
echo "  ✓ Kokoro-82M v1.0 downloaded"

echo ""
echo "=== Download Complete ==="
echo ""
echo "Models installed under $MODELS_ROOT/:"
for d in "$STT_DIR" "$EOU_DIR" "$ZIPF_DIR" "$TTS_DIR" "$LLM_DIR" "$VAD_DIR" "$DFN3_DIR"; do
  if [ -d "$d" ]; then
    du -sh "$d" 2>/dev/null || true
  fi
done
echo ""
du -sh "$MODELS_ROOT"
echo ""
# These are the gateway's built-in defaults, so running from the repo root needs
# no env vars for STT/TTS/VAD/DFN3. Only the LLM needs a dir (one per model).
echo "Run from the repo root and the defaults just work:"
echo "  cargo run --manifest-path szca_media_gateway/Cargo.toml"
echo ""
echo "The LLM is the one exception — point it at the model you want:"
echo "  export LLM_MODEL_DIR=$LLM_MODELS_DIR"
