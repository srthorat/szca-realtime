# SZCA Inference Service — real ONNX inference (STT/LLM/TTS/pipeline).
#
# Multi-stage: build a venv with all deps + the spaCy model Misaki needs baked
# in (so the runtime image needs NO network access), then copy it into a slim
# runtime image that runs as a non-root user.
#
# Models are NOT baked in (~1.9 GB); mount them at /models/engine (see
# docker-compose.yml). This keeps the image small and models out of the layer
# cache.

# ---- build stage -----------------------------------------------------------
FROM python:3.11-slim AS build

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# libsndfile is required by soundfile (PySoundFile) for WAV I/O.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt
# Bake the English spaCy model Misaki loads, so first request needs no network.
RUN python -m spacy download en_core_web_sm

# ---- runtime stage ---------------------------------------------------------
FROM python:3.11-slim AS runtime

# libsndfile1 (runtime lib only) + curl for the healthcheck.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libsndfile1 curl \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 szca

COPY --from=build /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    ENGINE_MODELS_DIR=/models/engine \
    LLM_MODELS_DIR=/models/engine/llm/Qwen2.5-1.5B-Instruct

WORKDIR /app
COPY app /app/app

USER szca
EXPOSE 8900

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8900/ready || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8900"]
