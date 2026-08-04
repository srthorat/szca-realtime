"""
Dev LLM server — OpenAI-compatible SSE endpoint using ONNX Runtime GenAI.

Runs on CPU, single-request, no batching. Same wire format as prod vLLM so
the Rust gateway's VllmClient works unchanged in dev and prod.

Usage:
    pip install onnxruntime-genai huggingface_hub fastapi uvicorn
    python download_model.py   # downloads INT4 ONNX model (~4.9 GB)
    uvicorn dev_server:app --host 0.0.0.0 --port 8080

The Rust gateway connects to this server with:
    LLM_BACKEND=vllm LLM_BASE_URL=http://localhost:8080
"""

import asyncio
import json
import os

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional

app = FastAPI(title="SZCA Dev LLM Server")

MODEL_PATH = os.environ.get(
    "LLM_MODEL_PATH",
    "./models/llm/llama3.1-8b-onnx/cpu_and_mobile/cpu-int4-rtn-block-32-acc-level-4",
)

# Lazy-loaded model (first request triggers load).
_model = None
_tokenizer = None


def _get_model():
    global _model, _tokenizer
    if _model is None:
        import onnxruntime_genai as og
        print(f"Loading model from {MODEL_PATH}...")
        _model = og.Model(MODEL_PATH)
        _tokenizer = og.Tokenizer(_model)
        print("Model loaded.")
    return _model, _tokenizer


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str = "llama-3.1-8b"
    messages: List[Message]
    max_tokens: int = 256
    stream: bool = True
    temperature: float = 0.7


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_PATH}


@app.post("/v1/chat/completions")
async def chat(req: ChatRequest):
    model, tokenizer = _get_model()

    prompt = tokenizer.apply_chat_template(
        [m.model_dump() for m in req.messages], add_generation_prompt=True
    )
    tokens = tokenizer.encode(prompt)

    import onnxruntime_genai as og

    params = og.GeneratorParams(model)
    params.set_search_options(max_length=len(tokens) + req.max_tokens)
    params.input_ids = tokens
    generator = og.Generator(model, params)

    async def token_stream():
        while not generator.is_done():
            generator.compute_logits()
            generator.generate_next_token()
            token = generator.get_next_tokens()[0]
            text = tokenizer.decode([token])
            chunk = {
                "id": "dev-0",
                "object": "chat.completion.chunk",
                "choices": [{"delta": {"content": text}, "finish_reason": None}],
            }
            yield f"data: {json.dumps(chunk)}\n\n"
            await asyncio.sleep(0)
        done = {
            "id": "dev-0",
            "object": "chat.completion.chunk",
            "choices": [{"delta": {}, "finish_reason": "stop"}],
        }
        yield f"data: {json.dumps(done)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(token_stream(), media_type="text/event-stream")
