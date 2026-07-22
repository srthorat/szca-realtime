# syntax=docker/dockerfile:1
# ============================================================================
# SZCA GPU image (H14 hardened, multi-stage)
# ----------------------------------------------------------------------------
# Base images are pinned to specific patch tags. For stronger supply-chain
# guarantees pin by immutable digest instead.
# TODO: pin by digest, e.g.
#   FROM rust:1.75-slim-bookworm@sha256:<digest> as gateway-builder
#   FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04@sha256:<digest>
# ============================================================================

# ---- Stage 1: build the Rust media gateway --------------------------------
FROM rust:1.75-slim-bookworm as gateway-builder

WORKDIR /app
COPY szca_media_gateway/ .
RUN cargo build --release

# ---- Stage 2: build the C++ ONNX engine -----------------------------------
FROM ubuntu:22.04 as engine-builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake \
    libonnxruntime-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY szca_onnx_engine/ .
RUN mkdir -p build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Release && make -j"$(nproc)"

# ---- Stage 3: runtime image (no build toolchains) -------------------------
FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04

# Runtime needs the ONNX Runtime shared library (the runtime package, NOT the
# -dev package used in the build stage) plus curl for the HEALTHCHECK. Build
# tools (gcc, cmake, cargo) are intentionally absent from this stage.
# NOTE: the exact runtime package name depends on how ONNX Runtime is provided
# to your apt sources (PPA / vendored .deb). Adjust `libonnxruntime` below to
# match the runtime (non -dev) package available in your build environment.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libonnxruntime \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Create an unprivileged user to run the service.
RUN groupadd --system appuser && useradd --system --gid appuser --create-home appuser

# Copy the release binaries from the build stages.
COPY --from=gateway-builder /app/target/release/szca_media_gateway /usr/local/bin/szca_media_gateway
COPY --from=engine-builder  /app/build/szca_onnx_engine            /usr/local/bin/szca_onnx_engine

# Application working directory owned by the non-root user.
WORKDIR /app
RUN chown -R appuser:appuser /app

# Models are NOT baked into the image. Mount them as a volume at runtime, e.g.
#   docker run -v $(pwd)/szca_media_gateway/models:/models szca-gpu:latest
# (Run ./download_models.sh on the host first.)
VOLUME ["/models"]

USER appuser

EXPOSE 3000

# The gateway listens on 0.0.0.0:3000 and serves GET /health.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD curl --fail http://localhost:3000/health || exit 1

# Reference the binary by an absolute path that exists in the image.
CMD ["/usr/local/bin/szca_media_gateway"]
