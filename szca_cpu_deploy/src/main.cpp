/// SZCA CPU-Only Deployment — Entry Point
///
/// Lightweight inference engine for testing without GPU.
/// Uses llama.cpp for LLM, whisper.cpp for STT, ONNX CPU for TTS.

#include "engine.h"
#include "http_server.h"
#include <iostream>
#include <signal.h>
#include <atomic>
#include <thread>
#include <chrono>
#include <string>
#include <cstring>
#include <stdexcept>

static std::atomic<bool> running(true);

static void signal_handler(int sig) {
    running.store(false);
}

void print_banner() {
    std::cout << "╔══════════════════════════════════════════════════════════════╗" << std::endl;
    std::cout << "║  SZCA CPU-Only Engine v5.0.0                                ║" << std::endl;
    std::cout << "║  No GPU required — Pure C++ CPU inference                   ║" << std::endl;
    std::cout << "╠══════════════════════════════════════════════════════════════╣" << std::endl;
    std::cout << "║  STT: Parakeet TDT 0.6B (whisper.cpp)                       ║" << std::endl;
    std::cout << "║  LLM: Hermes-3 3B INT4 (llama.cpp)                          ║" << std::endl;
    std::cout << "║  TTS: Kokoro-82M (ONNX CPU)                                 ║" << std::endl;
    std::cout << "║  Audio: 16kHz PCM 16-bit Mono                               ║" << std::endl;
    std::cout << "╚══════════════════════════════════════════════════════════════╝" << std::endl;
}

int main(int argc, char* argv[]) {
    print_banner();

    // Parse arguments
    int port = 8080;
    std::string model_dir = "./models/";

    auto usage = [&]() {
        std::cerr << "Usage: " << argv[0]
                  << " [--port <1-65535>] [--model_dir <path>]" << std::endl;
    };

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--port") {
            if (i + 1 >= argc) {  // L6: guard missing argument
                std::cerr << "[SZCA] --port requires a value" << std::endl;
                usage();
                return 1;
            }
            // L6: std::stoi throws on non-numeric / out-of-range input.
            try {
                size_t consumed = 0;
                int parsed = std::stoi(argv[i + 1], &consumed);
                if (consumed != std::strlen(argv[i + 1]) ||
                    parsed < 1 || parsed > 65535) {
                    throw std::out_of_range("port range");
                }
                port = parsed;
            } catch (const std::exception&) {
                std::cerr << "[SZCA] Invalid port: " << argv[i + 1]
                          << " (must be 1..65535)" << std::endl;
                usage();
                return 1;
            }
            ++i;
        } else if (arg == "--model_dir") {
            if (i + 1 >= argc) {
                std::cerr << "[SZCA] --model_dir requires a value" << std::endl;
                usage();
                return 1;
            }
            model_dir = argv[++i];
        } else {
            std::cerr << "[SZCA] Unknown argument: " << arg << std::endl;
            usage();
            return 1;
        }
    }

    // Initialize engine
    szca::EngineConfig config;
    config.stt.model_path = model_dir + "parakeet_tdt_0.6b_v3_fp16.onnx";
    config.llm.model_path = model_dir + "hermes-3-3b-int4.gguf";
    config.tts.model_path = model_dir + "kokoro_v1.0.onnx";
    config.tts.voices_path = model_dir + "voices.bin";

    szca::Engine engine(config);

    if (!engine.initialize()) {
        std::cerr << "[SZCA] Failed to initialize engine" << std::endl;
        return 1;
    }

    std::cout << "[SZCA] Engine initialized" << std::endl;

    // Start HTTP server
    szca::HttpServer server(port, &engine);

    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);

    std::cout << "[SZCA] Listening on http://0.0.0.0:" << port << std::endl;
    std::cout << "[SZCA] Endpoints:" << std::endl;
    std::cout << "  POST /v1/stt/stream    — Streaming STT" << std::endl;
    std::cout << "  POST /v1/llm/stream    — Streaming LLM" << std::endl;
    std::cout << "  POST /v1/tts/stream    — Streaming TTS" << std::endl;
    std::cout << "  POST /v1/voice         — Unified Voice API" << std::endl;
    std::cout << "  GET  /health           — Health check" << std::endl;
    std::cout << "" << std::endl;
    std::cout << "[SZCA] Press Ctrl+C to stop" << std::endl;

    server.start();

    while (running.load()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    server.stop();
    std::cout << "[SZCA] Shutdown complete" << std::endl;

    return 0;
}
