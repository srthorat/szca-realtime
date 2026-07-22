/// Resampler module unit tests.

#include "resampler.h"
#include <cassert>
#include <iostream>
#include <cmath>

using namespace szca;

void test_resampler_config_default() {
    ResamplerConfig config;
    assert(config.input_sample_rate == 24000);
    assert(config.output_sample_rate == 16000);
    assert(config.channels == 1);
    std::cout << "  [PASS] test_resampler_config_default" << std::endl;
}

void test_resampler_new() {
    ResamplerConfig config;
    Resampler resampler(config);
    assert(!resampler.is_initialized());
    std::cout << "  [PASS] test_resampler_new" << std::endl;
}

void test_resampler_initialize() {
    ResamplerConfig config;
    Resampler resampler(config);
    assert(resampler.initialize());
    assert(resampler.is_initialized());
    std::cout << "  [PASS] test_resampler_initialize" << std::endl;
}

void test_resampler_output_sample_count() {
    ResamplerConfig config;
    Resampler resampler(config);
    // 24000 → 16000 = 2/3 ratio
    assert(resampler.output_sample_count(300) == 200);
    assert(resampler.output_sample_count(480) == 320);
    std::cout << "  [PASS] test_resampler_output_sample_count" << std::endl;
}

void test_resampler_process_silence() {
    ResamplerConfig config;
    Resampler resampler(config);
    resampler.initialize();

    std::vector<int16_t> input(480, 0); // 20ms @ 24kHz
    auto output = resampler.process(input.data(), static_cast<int>(input.size()));
    assert(output.size() == 320); // 20ms @ 16kHz
    for (auto sample : output) {
        assert(sample == 0);
    }
    std::cout << "  [PASS] test_resampler_process_silence" << std::endl;
}

void test_resampler_process_sine_wave() {
    ResamplerConfig config;
    Resampler resampler(config);
    resampler.initialize();

    // Generate 440Hz sine wave at 24kHz
    std::vector<int16_t> input(480);
    for (int i = 0; i < 480; i++) {
        double t = static_cast<double>(i) / 24000.0;
        input[i] = static_cast<int16_t>(1000.0 * sin(2.0 * M_PI * 440.0 * t));
    }

    auto output = resampler.process(input.data(), static_cast<int>(input.size()));
    assert(output.size() == 320);

    // Check output is not all zeros
    bool has_nonzero = false;
    for (auto sample : output) {
        if (sample != 0) { has_nonzero = true; break; }
    }
    assert(has_nonzero);
    std::cout << "  [PASS] test_resampler_process_sine_wave" << std::endl;
}

void test_resampler_process_not_initialized() {
    ResamplerConfig config;
    Resampler resampler(config);
    std::vector<int16_t> input(480, 100);
    auto output = resampler.process(input.data(), static_cast<int>(input.size()));
    assert(output.empty());
    std::cout << "  [PASS] test_resampler_process_not_initialized" << std::endl;
}

void test_resampler_process_null_input() {
    ResamplerConfig config;
    Resampler resampler(config);
    resampler.initialize();
    auto output = resampler.process(nullptr, 0);
    assert(output.empty());
    std::cout << "  [PASS] test_resampler_process_null_input" << std::endl;
}

void test_resampler_config_accessor() {
    ResamplerConfig config;
    config.input_sample_rate = 48000;
    Resampler resampler(config);
    assert(resampler.config().input_sample_rate == 48000);
    std::cout << "  [PASS] test_resampler_config_accessor" << std::endl;
}

void test_resampler_preserves_amplitude() {
    ResamplerConfig config;
    Resampler resampler(config);
    resampler.initialize();

    // Constant amplitude signal
    std::vector<int16_t> input(480, 1000);
    auto output = resampler.process(input.data(), static_cast<int>(input.size()));

    // Check output amplitude is close to input
    for (auto sample : output) {
        assert(std::abs(sample - 1000) < 50); // Allow small interpolation error
    }
    std::cout << "  [PASS] test_resampler_preserves_amplitude" << std::endl;
}

int main() {
    std::cout << "Running Resampler tests..." << std::endl;
    test_resampler_config_default();
    test_resampler_new();
    test_resampler_initialize();
    test_resampler_output_sample_count();
    test_resampler_process_silence();
    test_resampler_process_sine_wave();
    test_resampler_process_not_initialized();
    test_resampler_process_null_input();
    test_resampler_config_accessor();
    test_resampler_preserves_amplitude();
    std::cout << "Resampler tests: ALL PASSED" << std::endl;
    return 0;
}
