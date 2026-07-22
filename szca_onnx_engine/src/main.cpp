/// SZCA ONNX Inference Engine — Entry Point
///
/// Unified C++ inference engine running:
/// - Parakeet TDT 0.6B (STT)
/// - Hermes-3 3B INT8 (LLM)
/// - Kokoro-82M (TTS)
/// - SoXR Resampler (24k → 16k)

#include "engine.h"
#include <iostream>
#include <signal.h>
#include <atomic>
#include <thread>
#include <chrono>

static std::atomic<bool> running(true);

static void signal_handler(int sig) {
    running.store(false);
}

int main(int argc, char* argv[]) {
    std::cout << "┌─────────────────────────────────────────────────────┐" << std::endl;
    std::cout << "│  SZCA ONNX Inference Engine v5.0.0                  │" << std::endl;
    std::cout << "│  STT: Parakeet TDT 0.6B V3 (CC-BY-4.0)             │" << std::endl;
    std::cout << "│  LLM: Hermes-3-Llama-3.2-3B INT8 (Apache 2.0)      │" << std::endl;
    std::cout << "│  TTS: Kokoro-82M (MIT)                              │" << std::endl;
    std::cout << "└─────────────────────────────────────────────────────┘" << std::endl;

    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);

    szca::EngineConfig config;
    szca::Engine engine(config);

    if (!engine.initialize()) {
        std::cerr << "[SZCA] Failed to initialize engine" << std::endl;
        return 1;
    }

    std::cout << "[SZCA] Engine initialized successfully" << std::endl;
    std::cout << "[SZCA] Waiting for audio input..." << std::endl;

    while (running.load()) {
        // In production: read from IPC channel, process, write back
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    std::cout << "[SZCA] Engine shutting down" << std::endl;
    return 0;
}
