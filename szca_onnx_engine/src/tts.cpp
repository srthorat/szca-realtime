/// TTS (Text-to-Speech) module implementation.
///
/// Kokoro-82M ONNX streaming speech synthesis.

#include "tts.h"
#include "ort_utils.h"
#include <cmath>
#include <iostream>
#include <fstream>

namespace szca {

struct TtsProcessor::Impl {
    OrtSessionWrapper session;
    std::vector<std::string> voices;
    std::vector<float> voice_embeddings;
};

TtsProcessor::TtsProcessor(const TtsConfig& config)
    : config_(config)
    , initialized_(false)
    , voices_()
    , impl_(std::make_unique<Impl>()) {
}

TtsProcessor::~TtsProcessor() = default;

bool TtsProcessor::initialize() {
    if (!impl_->session.load(config_.model_path)) {
        std::cerr << "[TTS] Failed to load model: " << config_.model_path << std::endl;
        return false;
    }

    // Load voice embeddings from voices.bin
    std::ifstream voice_file(config_.voices_path, std::ios::binary);
    if (voice_file.is_open()) {
        // Read voice embeddings
        voice_file.seekg(0, std::ios::end);
        std::streamoff raw_size = voice_file.tellg();
        voice_file.seekg(0, std::ios::beg);

        // M12: tellg() returns -1 on failure; guard before using as a size.
        if (raw_size < 0) {
            std::cerr << "[TTS] Failed to determine voices file size: "
                      << config_.voices_path << std::endl;
        } else if (static_cast<size_t>(raw_size) % sizeof(float) != 0) {
            std::cerr << "[TTS] voices file size not a multiple of float size: "
                      << raw_size << " bytes" << std::endl;
        } else {
            size_t file_size = static_cast<size_t>(raw_size);
            impl_->voice_embeddings.resize(file_size / sizeof(float));
            voice_file.read(
                reinterpret_cast<char*>(impl_->voice_embeddings.data()),
                static_cast<std::streamsize>(file_size)
            );
            if (!voice_file ||
                static_cast<size_t>(voice_file.gcount()) != file_size) {
                std::cerr << "[TTS] Failed to read voice embeddings" << std::endl;
                impl_->voice_embeddings.clear();
            } else {
                std::cout << "[TTS] Loaded voice embeddings: " << file_size
                          << " bytes" << std::endl;
            }
        }
        voice_file.close();
    }

    // Default voice list
    impl_->voices = {
        "af_heart", "af_bella", "af_nicole", "af_sarah", "af_sky",
        "am_adam", "am_michael",
        "bf_emma", "bf_isabella",
        "bm_george", "bm_lewis"
    };

    initialized_ = true;
    std::cout << "[TTS] Initialized: " << config_.model_path << std::endl;
    return true;
}

std::vector<TtsChunk> TtsProcessor::synthesize(const std::string& text) {
    std::vector<TtsChunk> chunks;

    if (!initialized_ || text.empty()) {
        return chunks;
    }

    // In production: tokenize text, run Kokoro ONNX, decode audio
    // For now, generate silence placeholder
    int samples_per_chunk = config_.sample_rate * 20 / 1000; // 20ms chunks
    int total_chunks = std::max(1, static_cast<int>(text.size()) / 10);

    for (int i = 0; i < total_chunks; i++) {
        TtsChunk chunk;
        chunk.pcm.resize(samples_per_chunk, 0);
        chunk.sample_rate = config_.sample_rate;
        chunk.duration_ms = 20.0f;
        chunks.push_back(std::move(chunk));
    }

    return chunks;
}

bool TtsProcessor::is_initialized() const {
    return initialized_;
}

const TtsConfig& TtsProcessor::config() const {
    return config_;
}

std::vector<std::string> TtsProcessor::available_voices() const {
    return impl_->voices;
}

} // namespace szca
