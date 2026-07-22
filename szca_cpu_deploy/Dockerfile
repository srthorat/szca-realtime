FROM ubuntu:22.04

# Install dependencies
RUN apt-get update && apt-get install -y \
    build-essential cmake \
    libcurl4-openssl-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy source files
COPY include/ include/
COPY src/ src/
COPY tests/ tests/
COPY CMakeLists.txt .

# Build
RUN mkdir -p build && cd build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release && \
    make -j$(nproc)

# Copy models
COPY models/ /models/

EXPOSE 8080

CMD ["./build/szca_cpu", "--port", "8080", "--model_dir", "/models/"]
