/// STT module unit tests.

#include "stt.h"
#include <cassert>
#include <iostream>
#include <vector>
#include <cstring>

using namespace szca;

// Helper to create audio chunk
static std::vector<int16_t> make_silence(int samples) {
    return std::vector<int16_t>(samples, 0);
}

static std::vector<int16_t> make_speech(int samples, int16_t amplitude = 3000) {
    std::vector<int16_t> audio(samples);
    for (int i = 0; i < samples; i++) {
        double t = static_cast<double>(i) / 16000.0;
        double angle = 2.0 * M_PI * 440.0 * t;
        audio[i] = static_cast<int16_t>(amplitude * (sin(angle) >= 0 ? 1 : -1));
    }
    return audio;
}

void test_stt_config_default() {
    SttConfig config;
    assert(config.model_path.find("parakeet") != std::string::npos);
    assert(config.sample_rate == 16000);
    assert(config.chunk_duration_ms == 20);
    assert(config.language == "en");
    assert(config.interim_results == true);
    std::cout << "  [PASS] test_stt_config_default" << std::endl;
}

void test_stt_processor_new() {
    SttConfig config;
    SttProcessor processor(config);
    assert(!processor.is_initialized());
    assert(processor.frame_count() == 0);
    std::cout << "  [PASS] test_stt_processor_new" << std::endl;
}

void test_stt_initialize() {
    SttConfig config;
    SttProcessor processor(config);
    assert(processor.initialize());
    assert(processor.is_initialized());
    std::cout << "  [PASS] test_stt_initialize" << std::endl;
}

void test_stt_process_not_initialized() {
    SttConfig config;
    SttProcessor processor(config);
    auto silence = make_silence(320);
    auto result = processor.process(silence.data(), static_cast<int>(silence.size()));
    assert(result == nullptr);
    std::cout << "  [PASS] test_stt_process_not_initialized" << std::endl;
}

void test_stt_process_silence() {
    SttConfig config;
    SttProcessor processor(config);
    processor.initialize();
    auto silence = make_silence(320);
    auto result = processor.process(silence.data(), static_cast<int>(silence.size()));
    assert(result == nullptr);
    assert(processor.frame_count() == 1);
    std::cout << "  [PASS] test_stt_process_silence" << std::endl;
}

void test_stt_process_speech() {
    SttConfig config;
    SttProcessor processor(config);
    processor.initialize();
    auto speech = make_speech(320, 3000);
    auto result = processor.process(speech.data(), static_cast<int>(speech.size()));
    assert(result != nullptr);
    assert(result->type == SttResult::Partial);
    assert(result->confidence > 0.0f);
    assert(processor.frame_count() == 1);
    std::cout << "  [PASS] test_stt_process_speech" << std::endl;
}

void test_stt_process_multiple_frames() {
    SttConfig config;
    SttProcessor processor(config);
    processor.initialize();

    auto speech = make_speech(320, 3000);
    for (int i = 0; i < 10; i++) {
        processor.process(speech.data(), static_cast<int>(speech.size()));
    }
    assert(processor.frame_count() == 10);
    std::cout << "  [PASS] test_stt_process_multiple_frames" << std::endl;
}

void test_stt_silence_flushes_final() {
    SttConfig config;
    SttProcessor processor(config);
    processor.initialize();

    // Speech first
    auto speech = make_speech(320, 3000);
    auto result = processor.process(speech.data(), static_cast<int>(speech.size()));
    assert(result != nullptr);

    // Then silence — should flush final
    auto silence = make_silence(320);
    result = processor.process(silence.data(), static_cast<int>(silence.size()));
    assert(result != nullptr);
    assert(result->type == SttResult::Final);
    std::cout << "  [PASS] test_stt_silence_flushes_final" << std::endl;
}

void test_stt_config_accessor() {
    SttConfig config;
    config.language = "es";
    SttProcessor processor(config);
    assert(processor.config().language == "es");
    std::cout << "  [PASS] test_stt_config_accessor" << std::endl;
}

void test_stt_no_interim_results() {
    SttConfig config;
    config.interim_results = false;
    SttProcessor processor(config);
    processor.initialize();

    auto speech = make_speech(320, 3000);
    auto result = processor.process(speech.data(), static_cast<int>(speech.size()));
    assert(result == nullptr); // No interim results
    std::cout << "  [PASS] test_stt_no_interim_results" << std::endl;
}

void test_stt_null_input() {
    SttConfig config;
    SttProcessor processor(config);
    processor.initialize();
    auto result = processor.process(nullptr, 0);
    assert(result == nullptr);
    std::cout << "  [PASS] test_stt_null_input" << std::endl;
}

void test_stt_timestamp_increments() {
    SttConfig config;
    SttProcessor processor(config);
    processor.initialize();

    auto speech = make_speech(320, 3000);
    auto r1 = processor.process(speech.data(), static_cast<int>(speech.size()));
    auto r2 = processor.process(speech.data(), static_cast<int>(speech.size()));

    assert(r1 != nullptr && r2 != nullptr);
    assert(r2->timestamp_ms > r1->timestamp_ms);
    std::cout << "  [PASS] test_stt_timestamp_increments" << std::endl;
}

int main() {
    std::cout << "Running STT tests..." << std::endl;
    test_stt_config_default();
    test_stt_processor_new();
    test_stt_initialize();
    test_stt_process_not_initialized();
    test_stt_process_silence();
    test_stt_process_speech();
    test_stt_process_multiple_frames();
    test_stt_silence_flushes_final();
    test_stt_config_accessor();
    test_stt_no_interim_results();
    test_stt_null_input();
    test_stt_timestamp_increments();
    std::cout << "STT tests: ALL PASSED" << std::endl;
    return 0;
}
