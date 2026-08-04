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
FROM rust:1.92-slim-bookworm@sha256:f1f73538ebe623fd3673a35aff3df358ae1084c64c55646516e5b17b321b6c9b as gateway-builder

WORKDIR /app
COPY szca_media_gateway/ .
RUN cargo build --release

# ---- Stage 2: build the C++ ONNX engine -----------------------------------
FROM ubuntu:22.04@sha256:ed154460658e1935fea730e26365007421370ed1ecb5b63004b5030e46045d6a as engine-builder

ARG ORT_VERSION=1.22.0
ARG ORT_SHA256_AARCH64=bb76395092d150b52c7092dc6b8f2fe4d80f0f3bf0416d2f269193e347e24702
ARG ORT_SHA256_X64=8344d55f93d5bc5021ce342db50f62079daf39aaafb5d311a451846228be49b3

RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential cmake curl ca-certificates pkg-config \
    && rm -rf /var/lib/apt/lists/* \
    && case "$(dpkg --print-architecture)" in \
         arm64) ORT_ARCH=aarch64; ORT_SHA="$ORT_SHA256_AARCH64" ;; \
         amd64) ORT_ARCH=x64;     ORT_SHA="$ORT_SHA256_X64" ;; \
         *) echo "unsupported arch: $(dpkg --print-architecture)" >&2; exit 1 ;; \
       esac \
    && ORT_TGZ="onnxruntime-linux-${ORT_ARCH}-${ORT_VERSION}.tgz" \
    && curl --fail --location --proto '=https' --tlsv1.2 -o /tmp/ort.tgz \
         "https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/${ORT_TGZ}" \
    && echo "${ORT_SHA}  /tmp/ort.tgz" | sha256sum --check --strict \
    && mkdir -p /opt/onnxruntime \
    && tar -xzf /tmp/ort.tgz -C /opt/onnxruntime --strip-components=1 \
    && rm /tmp/ort.tgz \
    && test -f /opt/onnxruntime/lib/libonnxruntime.so

WORKDIR /app
COPY szca_onnx_engine/ .
RUN mkdir -p build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Release && make -j"$(nproc)"

# ---- Stage 3: runtime image (no build toolchains) -------------------------
FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04@sha256:714a67503f269a8b1356fcf54972e2cf5e6a39b4b0e98031fa5a7ca8681121d5

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Copy pinned ONNX Runtime 1.22.0 from builder stage
COPY --from=engine-builder /opt/onnxruntime /opt/onnxruntime
ENV ORT_DYLIB_PATH=/opt/onnxruntime/lib/libonnxruntime.so
ENV LD_LIBRARY_PATH=/opt/onnxruntime/lib:${LD_LIBRARY_PATH}

# Create an unprivileged user to run the service.
RUN groupadd --system appuser && useradd --system --gid appuser --create-home appuser

# Copy the release binaries from the build stages.
COPY --from=gateway-builder /app/target/release/szca_media_gateway /usr/local/bin/szca_media_gateway
COPY --from=engine-builder  /app/build/szca_onnx_engine            /usr/local/bin/szca_onnx_engine

# Application working directory owned by the non-root user.
WORKDIR /app
RUN chown -R appuser:appuser /app

# Models are NOT baked into the image. Mount them as a volume at runtime, e.g.
#   docker run -v $(pwd)/models:/models szca-gpu:latest
# (Run ./download_models.sh on the host first.)
VOLUME ["/models"]

USER appuser

EXPOSE 3000

# The gateway listens on 0.0.0.0:3000 and serves GET /health.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD curl --fail http://localhost:3000/health || exit 1

# Reference the binary by an absolute path that exists in the image.
CMD ["/usr/local/bin/szca_media_gateway"]
