/// STT (Speech-to-Text) module header.
///
/// Parakeet TDT 0.6B V3 FP16 ONNX streaming transcription.
/// License: CC-BY-4.0 (credit NVIDIA)

#pragma once

#include <string>
#include <vector>
#include <cstdint>
#include <functional>
#include <memory>

namespace szca {

/// STT configuration.
struct SttConfig {
    std::string model_path = "./models/parakeet_tdt_0.6b_v3_fp16.onnx";
    int sample_rate = 16000;
    int chunk_duration_ms = 20;
    std::string language = "en";
    bool interim_results = true;
};

/// Word timing information.
struct WordTiming {
    std::string word;
    float start_ms;
    float end_ms;
    float confidence;
};

/// STT partial/final result.
struct SttResult {
    enum Type { Partial, Final };
    Type type;
    std::string text;
    float confidence;
    int64_t timestamp_ms;
    std::vector<WordTiming> words;
};

/// STT processor interface.
class SttProcessor {
public:
    explicit SttProcessor(const SttConfig& config);
    ~SttProcessor();

    // Non-copyable
    SttProcessor(const SttProcessor&) = delete;
    SttProcessor& operator=(const SttProcessor&) = delete;

    /// Initialize the processor (load model).
    bool initialize();

    /// Process an audio chunk and return transcription result.
    /// Returns nullptr if no result ready yet.
    std::unique_ptr<SttResult> process(const int16_t* pcm_data, int sample_count);

    /// Check if processor is initialized.
    bool is_initialized() const;

    /// Get configuration.
    const SttConfig& config() const;

    /// Get total frames processed.
    uint64_t frame_count() const;

private:
    SttConfig config_;
    bool initialized_;
    uint64_t frame_count_;
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace szca
