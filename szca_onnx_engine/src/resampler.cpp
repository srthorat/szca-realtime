/// Audio Resampler implementation.
///
/// SoXR-based high-quality audio resampling (24kHz → 16kHz).

#include "resampler.h"
#include <cmath>

namespace szca {

Resampler::Resampler(const ResamplerConfig& config)
    : config_(config)
    , initialized_(false) {
}

Resampler::~Resampler() = default;

bool Resampler::initialize() {
    // L1: validate sample rates to avoid divide-by-zero / NaN ratios below.
    if (config_.input_sample_rate <= 0 || config_.output_sample_rate <= 0) {
        return false;
    }
    // In production: initialize SoXR resampler.
    // QUALITY TODO: this uses simple linear interpolation which lacks an
    // anti-aliasing filter; when downsampling (e.g. 24kHz -> 16kHz) this can
    // introduce aliasing artifacts. A proper implementation should use SoXR
    // (or an equivalent polyphase FIR) for high-quality resampling.
    initialized_ = true;
    return true;
}

std::vector<int16_t> Resampler::process(const int16_t* input, int input_samples) {
    std::vector<int16_t> output;

    if (!initialized_ || !input || input_samples <= 0) {
        return output;
    }

    // Calculate output size
    double ratio = static_cast<double>(config_.output_sample_rate) /
                   static_cast<double>(config_.input_sample_rate);
    int output_samples = static_cast<int>(input_samples * ratio);
    output.resize(output_samples);

    // Simple linear interpolation resampler
    // In production: use SoXR for high quality
    for (int i = 0; i < output_samples; i++) {
        double src_idx = i / ratio;
        int idx = static_cast<int>(src_idx);
        double frac = src_idx - idx;

        if (idx + 1 < input_samples) {
            // Linear interpolation
            double sample = input[idx] * (1.0 - frac) + input[idx + 1] * frac;
            output[i] = static_cast<int16_t>(std::clamp(sample, -32768.0, 32767.0));
        } else if (idx < input_samples) {
            output[i] = input[idx];
        } else {
            output[i] = 0;
        }
    }

    return output;
}

bool Resampler::is_initialized() const {
    return initialized_;
}

const ResamplerConfig& Resampler::config() const {
    return config_;
}

int Resampler::output_sample_count(int input_samples) const {
    // L1: guard against divide-by-zero if rates are invalid.
    if (config_.input_sample_rate <= 0 || config_.output_sample_rate <= 0) {
        return 0;
    }
    double ratio = static_cast<double>(config_.output_sample_rate) /
                   static_cast<double>(config_.input_sample_rate);
    return static_cast<int>(input_samples * ratio);
}

} // namespace szca
