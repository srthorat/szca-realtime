/// Audio Resampler module header.
///
/// SoXR-based high-quality audio resampling.
/// Resamples TTS output (24kHz) to output format (16kHz).
/// License: LGPL-2.1

#pragma once

#include <vector>
#include <cstdint>

namespace szca {

/// Resampler configuration.
struct ResamplerConfig {
    int input_sample_rate = 24000;
    int output_sample_rate = 16000;
    int channels = 1;
    float quality = 0.95f;  // 0.0 = lowest, 1.0 = highest
};

/// Resampler processor.
class Resampler {
public:
    explicit Resampler(const ResamplerConfig& config);
    ~Resampler();

    Resampler(const Resampler&) = delete;
    Resampler& operator=(const Resampler&) = delete;

    /// Initialize the resampler.
    bool initialize();

    /// Resample audio data.
    std::vector<int16_t> process(const int16_t* input, int input_samples);

    /// Check if initialized.
    bool is_initialized() const;

    /// Get configuration.
    const ResamplerConfig& config() const;

    /// Calculate output sample count for given input.
    int output_sample_count(int input_samples) const;

private:
    ResamplerConfig config_;
    bool initialized_;
};

} // namespace szca
