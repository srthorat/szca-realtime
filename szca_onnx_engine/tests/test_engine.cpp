/// Engine module unit tests.

#include "engine.h"
#include <cassert>
#include <iostream>
#include <vector>

using namespace szca;

void test_engine_config_default() {
    EngineConfig config;
    assert(config.stt.sample_rate == 16000);
    assert(config.llm.max_tokens == 256);
    assert(config.tts.sample_rate == 24000);
    assert(config.resampler.input_sample_rate == 24000);
    std::cout << "  [PASS] test_engine_config_default" << std::endl;
}

void test_engine_new() {
    EngineConfig config;
    Engine engine(config);
    assert(!engine.is_initialized());
    std::cout << "  [PASS] test_engine_new" << std::endl;
}

void test_engine_initialize() {
    EngineConfig config;
    Engine engine(config);
    assert(engine.initialize());
    assert(engine.is_initialized());
    std::cout << "  [PASS] test_engine_initialize" << std::endl;
}

void test_engine_process_audio_null() {
    EngineConfig config;
    Engine engine(config);
    engine.initialize();
    // Should not crash
    engine.process_audio(nullptr, 0, nullptr, nullptr);
    std::cout << "  [PASS] test_engine_process_audio_null" << std::endl;
}

void test_engine_process_audio_silence() {
    EngineConfig config;
    Engine engine(config);
    engine.initialize();

    std::vector<int16_t> silence(320, 0);
    int text_calls = 0;
    int audio_calls = 0;

    engine.process_audio(
        silence.data(),
        static_cast<int>(silence.size()),
        [&](const int16_t*, int, int) { audio_calls++; },
        [&](const std::string&, bool) { text_calls++; }
    );

    // Silence should not produce text or audio
    assert(audio_calls == 0);
    std::cout << "  [PASS] test_engine_process_audio_silence" << std::endl;
}

void test_engine_process_audio_speech() {
    EngineConfig config;
    Engine engine(config);
    engine.initialize();

    // Generate speech-like audio
    std::vector<int16_t> speech(320);
    for (int i = 0; i < 320; i++) {
        double t = static_cast<double>(i) / 16000.0;
        speech[i] = static_cast<int16_t>(3000.0 * (sin(2.0 * M_PI * 440.0 * t) >= 0 ? 1 : -1));
    }

    int text_calls = 0;
    engine.process_audio(
        speech.data(),
        static_cast<int>(speech.size()),
        nullptr,
        [&](const std::string&, bool) { text_calls++; }
    );

    // Speech should produce text
    assert(text_calls > 0);
    std::cout << "  [PASS] test_engine_process_audio_speech" << std::endl;
}

void test_engine_session() {
    EngineConfig config;
    Engine engine(config);
    engine.initialize();

    assert(engine.session().state() == SessionState::Created);
    engine.session().activate();
    assert(engine.session().state() == SessionState::Active);
    std::cout << "  [PASS] test_engine_session" << std::endl;
}

void test_engine_config_accessor() {
    EngineConfig config;
    config.llm.max_tokens = 512;
    Engine engine(config);
    assert(engine.config().llm.max_tokens == 512);
    std::cout << "  [PASS] test_engine_config_accessor" << std::endl;
}

void test_engine_reset() {
    EngineConfig config;
    Engine engine(config);
    engine.initialize();
    engine.session().activate();
    engine.reset();
    assert(engine.session().state() == SessionState::Created);
    std::cout << "  [PASS] test_engine_reset" << std::endl;
}

int main() {
    std::cout << "Running Engine tests..." << std::endl;
    test_engine_config_default();
    test_engine_new();
    test_engine_initialize();
    test_engine_process_audio_null();
    test_engine_process_audio_silence();
    test_engine_process_audio_speech();
    test_engine_session();
    test_engine_config_accessor();
    test_engine_reset();
    std::cout << "Engine tests: ALL PASSED" << std::endl;
    return 0;
}
