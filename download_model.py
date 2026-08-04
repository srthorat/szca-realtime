"""
Download INT4 ONNX Llama 3.1 8B model for local dev server.

~4.9 GB download. Stores at ./models/llm/llama3.1-8b-onnx/ by default — under the
repo's single model root (models/{stt,tts,llm,vad,dfn3}), one directory per LLM.

This is only for the OPTIONAL Python dev LLM server (dev_server.py, used with
LLM_BACKEND=vllm). The in-process ONNX path uses ./download_models.sh instead.

Usage:
    pip install huggingface_hub
    python download_model.py
"""

import os
from huggingface_hub import snapshot_download

MODEL_DIR = os.environ.get("MODEL_DIR", "./models/llm/llama3.1-8b-onnx")

snapshot_download(
    repo_id="microsoft/Llama-3.1-8B-Instruct-onnx",
    allow_patterns=["cpu_and_mobile/cpu-int4-rtn-block-32-acc-level-4/*"],
    local_dir=MODEL_DIR,
)

print(f"Model downloaded to {MODEL_DIR}")
