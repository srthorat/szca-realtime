/// TTS (Text-to-Speech) module header.
///
/// Kokoro-82M ONNX streaming speech synthesis.
/// License: MIT

#pragma once

#include <string>
#include <vector>
#include <cstdint>
#include <memory>

namespace szca {

/// TTS configuration.
struct TtsConfig {
    std::string model_path = "./models/kokoro_v1.0.onnx";
    std::string voices_path = "./models/voices.bin";
    std::string voice = "af_heart";
    std::string language = "en-us";
    int sample_rate = 24000;  // Kokoro outputs 24kHz
    float speed = 1.0f;
};

/// TTS audio chunk.
struct TtsChunk {
    std::vector<int16_t> pcm;
    int sample_rate;
    float duration_ms;
};

/// TTS processor interface.
class TtsProcessor {
public:
    explicit TtsProcessor(const TtsConfig& config);
    ~TtsProcessor();

    TtsProcessor(const TtsProcessor&) = delete;
    TtsProcessor& operator=(const TtsProcessor&) = delete;

    /// Initialize the processor (load model + voices).
    bool initialize();

    /// Generate audio from text.
    /// Returns audio chunks as they are generated.
    std::vector<TtsChunk> synthesize(const std::string& text);

    /// Check if processor is initialized.
    bool is_initialized() const;

    /// Get configuration.
    const TtsConfig& config() const;

    /// Get available voices.
    std::vector<std::string> available_voices() const;

private:
    TtsConfig config_;
    bool initialized_;
    std::vector<std::string> voices_;
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace szca
