/// TTS module unit tests.

#include "tts.h"
#include <cassert>
#include <iostream>

using namespace szca;

void test_tts_config_default() {
    TtsConfig config;
    assert(config.model_path.find("kokoro") != std::string::npos);
    assert(config.voice == "af_heart");
    assert(config.language == "en-us");
    assert(config.sample_rate == 24000);
    assert(config.speed == 1.0f);
    std::cout << "  [PASS] test_tts_config_default" << std::endl;
}

void test_tts_processor_new() {
    TtsConfig config;
    TtsProcessor processor(config);
    assert(!processor.is_initialized());
    std::cout << "  [PASS] test_tts_processor_new" << std::endl;
}

void test_tts_initialize() {
    TtsConfig config;
    TtsProcessor processor(config);
    assert(processor.initialize());
    assert(processor.is_initialized());
    std::cout << "  [PASS] test_tts_initialize" << std::endl;
}

void test_tts_synthesize_empty() {
    TtsConfig config;
    TtsProcessor processor(config);
    processor.initialize();
    auto chunks = processor.synthesize("");
    assert(chunks.empty());
    std::cout << "  [PASS] test_tts_synthesize_empty" << std::endl;
}

void test_tts_synthesize_text() {
    TtsConfig config;
    TtsProcessor processor(config);
    processor.initialize();
    auto chunks = processor.synthesize("Hello world");
    assert(!chunks.empty());
    assert(chunks[0].sample_rate == 24000);
    assert(chunks[0].duration_ms == 20.0f);
    std::cout << "  [PASS] test_tts_synthesize_text" << std::endl;
}

void test_tts_synthesize_multiple_chunks() {
    TtsConfig config;
    TtsProcessor processor(config);
    processor.initialize();
    auto chunks = processor.synthesize("This is a longer sentence that should produce multiple audio chunks");
    assert(chunks.size() > 1);
    std::cout << "  [PASS] test_tts_synthesize_multiple_chunks" << std::endl;
}

void test_tts_available_voices() {
    TtsConfig config;
    TtsProcessor processor(config);
    processor.initialize();
    auto voices = processor.available_voices();
    assert(!voices.empty());
    assert(voices.size() > 5);
    std::cout << "  [PASS] test_tts_available_voices" << std::endl;
}

void test_tts_synthesize_not_initialized() {
    TtsConfig config;
    TtsProcessor processor(config);
    auto chunks = processor.synthesize("Hello");
    assert(chunks.empty());
    std::cout << "  [PASS] test_tts_synthesize_not_initialized" << std::endl;
}

void test_tts_config_accessor() {
    TtsConfig config;
    config.voice = "am_adam";
    TtsProcessor processor(config);
    assert(processor.config().voice == "am_adam");
    std::cout << "  [PASS] test_tts_config_accessor" << std::endl;
}

int main() {
    std::cout << "Running TTS tests..." << std::endl;
    test_tts_config_default();
    test_tts_processor_new();
    test_tts_initialize();
    test_tts_synthesize_empty();
    test_tts_synthesize_text();
    test_tts_synthesize_multiple_chunks();
    test_tts_available_voices();
    test_tts_synthesize_not_initialized();
    test_tts_config_accessor();
    std::cout << "TTS tests: ALL PASSED" << std::endl;
    return 0;
}
