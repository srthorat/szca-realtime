"""Shared engine holder: loads each model once, guards CPU concurrency.

Sessions are loaded a single time at startup and shared across all requests
(ONNX Runtime `session.run` is thread-safe and releases the GIL during compute,
so concurrent requests genuinely parallelize in ORT's native thread pool).

A bounded semaphore caps how many heavy inferences run at once so a burst of
requests cannot oversubscribe the CPU and collapse tail latency; excess
requests wait, and the API layer applies a timeout + 503 backpressure.
"""

from __future__ import annotations

import os
import threading

from .stt import SttEngine
from .llm import LlmEngine
from .tts import TtsEngine


class InferenceEngine:
    def __init__(self):
        # Models live under ONE root, split per stage: models/{stt,tts,llm}.
        # ENGINE_MODELS_DIR names that root; each stage dir can still be
        # overridden individually (e.g. to point the LLM at a specific
        # checkpoint directory, since every LLM gets its own).
        engine_models = os.environ.get("ENGINE_MODELS_DIR", "/models/engine")
        stt_models = os.environ.get("STT_MODELS_DIR", os.path.join(engine_models, "stt"))
        tts_models = os.environ.get("TTS_MODELS_DIR", os.path.join(engine_models, "tts"))
        llm_models = os.environ.get("LLM_MODELS_DIR", os.path.join(engine_models, "llm"))
        voice = os.environ.get("TTS_VOICE", "af_heart")
        # 0 => let ORT pick; override per-deployment.
        threads = int(os.environ.get("ORT_INTRA_OP_THREADS", "0"))
        # Cap concurrent heavy inferences (defaults to CPU count).
        max_conc = int(os.environ.get("MAX_CONCURRENCY", str(os.cpu_count() or 4)))

        self.stt = SttEngine(stt_models, intra_op_threads=threads)
        self.tts = TtsEngine(tts_models, voice=voice, intra_op_threads=threads)
        self.llm = LlmEngine(llm_models, intra_op_threads=threads)

        self._sem = threading.BoundedSemaphore(max_conc)
        self.max_concurrency = max_conc
        self.ready = True

    def acquire(self, timeout: float) -> bool:
        """Try to reserve an inference slot; False if the pool is saturated."""
        return self._sem.acquire(timeout=timeout)

    def release(self):
        self._sem.release()
