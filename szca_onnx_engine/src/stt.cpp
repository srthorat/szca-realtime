/// STT (Speech-to-Text) module implementation.
///
/// Parakeet TDT 0.6B V3 FP16 ONNX streaming transcription.

#include "stt.h"
#include "ort_utils.h"
#include <cstring>
#include <cmath>
#include <iostream>

namespace szca {

struct SttProcessor::Impl {
    OrtSessionWrapper session;
    std::string buffer;
};

SttProcessor::SttProcessor(const SttConfig& config)
    : config_(config)
    , initialized_(false)
    , frame_count_(0)
    , impl_(std::make_unique<Impl>()) {
}

SttProcessor::~SttProcessor() = default;

bool SttProcessor::initialize() {
    if (!impl_->session.load(config_.model_path)) {
        std::cerr << "[STT] Failed to load model: " << config_.model_path << std::endl;
        return false;
    }
    initialized_ = true;
    std::cout << "[STT] Initialized: " << config_.model_path << std::endl;
    return true;
}

std::unique_ptr<SttResult> SttProcessor::process(const int16_t* pcm_data, int sample_count) {
    // L16: bound the sample count to avoid unbounded allocation from a
    // malicious/garbage length. 16MB of samples (~500s @ 16kHz) is far beyond
    // any single streaming chunk.
    static constexpr int MAX_SAMPLES = 16 * 1024 * 1024;

    if (!initialized_ || !pcm_data || sample_count <= 0 ||
        sample_count > MAX_SAMPLES) {
        return nullptr;
    }

    frame_count_++;

    // Convert int16 to float for ONNX input
    std::vector<float> float_samples(sample_count);
    for (int i = 0; i < sample_count; i++) {
        float_samples[i] = static_cast<float>(pcm_data[i]) / 32768.0f;
    }

    // Prepare input tensor: [1, sample_count]
    std::vector<std::string> input_names = {"audio_signal"};
    std::vector<std::vector<int64_t>> input_shapes = {{1, sample_count}};
    std::vector<std::vector<float>> input_data = {float_samples};

    // Run inference
    auto outputs = impl_->session.run(input_names, input_shapes, input_data);

    if (outputs.empty() || outputs[0].empty()) {
        return nullptr;
    }

    // Check speech energy for VAD-like behavior
    double sum_squares = 0.0;
    for (int i = 0; i < sample_count; i++) {
        sum_squares += static_cast<double>(pcm_data[i]) * pcm_data[i];
    }
    double rms = std::sqrt(sum_squares / sample_count);

    const double SPEECH_THRESHOLD = 500.0;

    if (rms < SPEECH_THRESHOLD) {
        // Silence — flush buffered text as final
        if (!impl_->buffer.empty()) {
            auto result = std::make_unique<SttResult>();
            result->type = SttResult::Final;
            result->text = impl_->buffer;
            result->confidence = 0.9f;
            result->timestamp_ms = frame_count_ * config_.chunk_duration_ms;
            impl_->buffer.clear();
            return result;
        }
        return nullptr;
    }

    // Speech detected — extract partial text from model output
    // In production, decode tokens from model output
    std::string partial_text = "speech_frame_" + std::to_string(frame_count_);

    if (config_.interim_results) {
        auto result = std::make_unique<SttResult>();
        result->type = SttResult::Partial;
        result->text = partial_text;
        result->confidence = 0.85f;
        result->timestamp_ms = frame_count_ * config_.chunk_duration_ms;
        impl_->buffer = partial_text;
        return result;
    }

    return nullptr;
}

bool SttProcessor::is_initialized() const {
    return initialized_;
}

const SttConfig& SttProcessor::config() const {
    return config_;
}

uint64_t SttProcessor::frame_count() const {
    return frame_count_;
}

} // namespace szca
