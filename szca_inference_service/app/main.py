"""SZCA real-inference HTTP service (Python reference implementation).

Four endpoints, all backed by genuine ONNX inference (no stubs):

  POST /v1/stt        multipart audio file            -> {text}
  POST /v1/llm        {messages|prompt, ...}          -> {reply, tokens}
  POST /v1/tts        {text, speed}                   -> audio/wav
  POST /v1/pipeline   multipart audio (voice->voice)  -> audio/wav (+ headers)

Concurrency model: blocking ONNX work runs in a threadpool so the event loop
stays responsive; a bounded semaphore caps simultaneous heavy inferences and
returns 503 when saturated (backpressure) instead of melting tail latency.

This service is the CORRECTNESS ORACLE for the Rust production port: every
endpoint here has been validated with real audio/text (TTS<->STT round-trip,
LLM KV-cache generation), so the fast Rust path can be checked against it.
"""

from __future__ import annotations

import asyncio
import base64
import os
from typing import List, Optional

from fastapi import FastAPI, File, HTTPException, Response, UploadFile
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel

from .audio import decode_wav, encode_wav, resample
from .engine import InferenceEngine

app = FastAPI(title="SZCA Inference Service", version="1.0.0")

ENGINE: Optional[InferenceEngine] = None
ACQUIRE_TIMEOUT = float(os.environ.get("ACQUIRE_TIMEOUT_S", "5.0"))
MAX_UPLOAD_BYTES = int(os.environ.get("MAX_UPLOAD_BYTES", str(25 * 1024 * 1024)))


@app.on_event("startup")
def _startup():
    global ENGINE
    ENGINE = InferenceEngine()


class Message(BaseModel):
    role: str
    content: str


class LlmRequest(BaseModel):
    messages: Optional[List[Message]] = None
    prompt: Optional[str] = None
    max_new_tokens: int = 128
    temperature: float = 0.0
    top_p: float = 0.9
    top_k: Optional[int] = None
    repetition_penalty: Optional[float] = None


class TtsRequest(BaseModel):
    text: str
    speed: float = 1.0


def _require_engine() -> InferenceEngine:
    if ENGINE is None or not ENGINE.ready:
        raise HTTPException(status_code=503, detail="engine not ready")
    return ENGINE


async def _guarded(engine: InferenceEngine, fn, *args):
    """Run a blocking inference under the concurrency semaphore + threadpool."""
    if not engine.acquire(ACQUIRE_TIMEOUT):
        raise HTTPException(status_code=503, detail="server busy, retry later")
    try:
        return await run_in_threadpool(fn, *args)
    finally:
        engine.release()


async def _read_upload(file: UploadFile) -> bytes:
    data = await file.read()
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="audio too large")
    if not data:
        raise HTTPException(status_code=400, detail="empty upload")
    return data


@app.get("/health")
def health():
    return {"status": "ok", "service": "szca-inference", "version": "1.0.0"}


@app.get("/ready")
def ready():
    if ENGINE is None or not ENGINE.ready:
        raise HTTPException(status_code=503, detail="not ready")
    return {"status": "ready", "max_concurrency": ENGINE.max_concurrency}


def _transcribe(engine: InferenceEngine, data: bytes) -> str:
    wav, sr = decode_wav(data)
    wav16 = resample(wav, sr, 16000)
    return engine.stt.transcribe(wav16, 16000)


@app.post("/v1/stt")
async def stt_endpoint(file: UploadFile = File(...)):
    engine = _require_engine()
    data = await _read_upload(file)
    text = await _guarded(engine, _transcribe, engine, data)
    return {"text": text}


@app.post("/v1/llm")
async def llm_endpoint(req: LlmRequest):
    engine = _require_engine()
    if req.messages:
        messages = [{"role": m.role, "content": m.content} for m in req.messages]
    elif req.prompt:
        messages = [{"role": "user", "content": req.prompt}]
    else:
        raise HTTPException(status_code=400, detail="messages or prompt required")

    def _gen():
        return engine.llm.generate(
            messages,
            max_new_tokens=req.max_new_tokens,
            temperature=req.temperature,
            top_p=req.top_p,
            top_k=req.top_k,
            repetition_penalty=req.repetition_penalty,
        )

    reply, n = await _guarded(engine, _gen)
    return {"reply": reply, "tokens": n}


@app.post("/v1/tts")
async def tts_endpoint(req: TtsRequest):
    engine = _require_engine()
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="text required")

    def _synth():
        wav = engine.tts.synthesize(req.text, speed=req.speed)
        return encode_wav(wav, engine.tts.sample_rate)

    wav_bytes = await _guarded(engine, _synth)
    return Response(content=wav_bytes, media_type="audio/wav")


@app.post("/v1/pipeline")
async def pipeline_endpoint(file: UploadFile = File(...)):
    """Full voice->voice: WAV in -> STT -> LLM -> TTS -> WAV out."""
    engine = _require_engine()
    data = await _read_upload(file)

    def _run():
        wav, sr = decode_wav(data)
        wav16 = resample(wav, sr, 16000)
        transcript = engine.stt.transcribe(wav16, 16000)
        reply, _ = engine.llm.generate(
            [{"role": "user", "content": transcript}],
            max_new_tokens=96,
            temperature=0.0,
        )
        out_wav = engine.tts.synthesize(reply, speed=1.0)
        return transcript, reply, encode_wav(out_wav, engine.tts.sample_rate)

    transcript, reply, wav_bytes = await _guarded(engine, _run)
    # Expose the intermediate text via headers so callers can inspect the chain.
    return Response(
        content=wav_bytes,
        media_type="audio/wav",
        headers={
            "X-Transcript": base64.b64encode(transcript.encode()).decode(),
            "X-Reply": base64.b64encode(reply.encode()).decode(),
        },
    )
