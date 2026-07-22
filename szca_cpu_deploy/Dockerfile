# SZCA CPU-only deployment image.
#
# IMPORTANT: build from the REPOSITORY ROOT, not from szca_cpu_deploy/, because
# CMakeLists.txt pulls the shared engine sources from the sibling
# ../szca_onnx_engine project:
#
#   docker build -f szca_cpu_deploy/Dockerfile -t szca-cpu .
#
# Models are NOT baked in — mount them at runtime:  -v /path/to/models:/models

# ---- build stage ----
# TODO: pin by digest (ubuntu:22.04@sha256:...) once resolvable in your registry.
FROM ubuntu:22.04 AS build

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake libcurl4-openssl-dev ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src
# Both source trees are needed (cpu_deploy references ../szca_onnx_engine).
COPY szca_cpu_deploy/ szca_cpu_deploy/
COPY szca_onnx_engine/ szca_onnx_engine/

RUN cmake -S szca_cpu_deploy -B /build -DCMAKE_BUILD_TYPE=Release \
    && cmake --build /build -j "$(nproc)"

# ---- runtime stage ----
FROM ubuntu:22.04 AS runtime

# Runtime deps only: libcurl (client helpers) + curl for HEALTHCHECK.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libcurl4 curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Non-root user.
RUN useradd --system --create-home --uid 10001 appuser

COPY --from=build /build/szca_cpu /usr/local/bin/szca_cpu

# Models are mounted here at runtime (never copied into the image).
VOLUME ["/models"]

USER appuser
EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -fsS http://localhost:8080/health || exit 1

ENTRYPOINT ["/usr/local/bin/szca_cpu"]
CMD ["--port", "8080", "--model_dir", "/models/"]
