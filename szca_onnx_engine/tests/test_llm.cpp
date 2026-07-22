/// LLM module unit tests.

#include "llm.h"
#include <cassert>
#include <iostream>

using namespace szca;

void test_llm_config_default() {
    LlmConfig config;
    assert(config.model_path.find("hermes") != std::string::npos);
    assert(config.max_tokens == 256);
    assert(config.temperature == 0.7f);
    assert(config.top_p == 0.9f);
    std::cout << "  [PASS] test_llm_config_default" << std::endl;
}

void test_llm_processor_new() {
    LlmConfig config;
    LlmProcessor processor(config);
    assert(!processor.is_initialized());
    assert(processor.token_count() == 0);
    std::cout << "  [PASS] test_llm_processor_new" << std::endl;
}

void test_llm_initialize() {
    LlmConfig config;
    LlmProcessor processor(config);
    assert(processor.initialize());
    assert(processor.is_initialized());
    std::cout << "  [PASS] test_llm_initialize" << std::endl;
}

void test_llm_generate_next() {
    LlmConfig config;
    LlmProcessor processor(config);
    processor.initialize();
    processor.add_message("user", "Hello");

    auto token = processor.generate_next();
    assert(!token.text.empty());
    assert(token.token_id > 0);
    assert(token.index >= 0);
    assert(processor.token_count() == 1);
    std::cout << "  [PASS] test_llm_generate_next" << std::endl;
}

void test_llm_generate_multiple() {
    LlmConfig config;
    LlmProcessor processor(config);
    processor.initialize();
    processor.add_message("user", "Hello");

    for (int i = 0; i < 5; i++) {
        auto token = processor.generate_next();
        assert(!token.text.empty());
    }
    assert(processor.token_count() == 5);
    std::cout << "  [PASS] test_llm_generate_multiple" << std::endl;
}

void test_llm_generate_complete() {
    LlmConfig config;
    config.max_tokens = 20;
    LlmProcessor processor(config);
    processor.initialize();
    processor.add_message("user", "Hello");

    auto completion = processor.generate_complete();
    assert(!completion.full_text.empty());
    assert(completion.total_tokens > 0);
    assert(completion.finish_reason == "stop");
    std::cout << "  [PASS] test_llm_generate_complete" << std::endl;
}

void test_llm_reset() {
    LlmConfig config;
    LlmProcessor processor(config);
    processor.initialize();
    processor.add_message("user", "Hello");
    processor.generate_next();

    processor.reset();
    assert(processor.token_count() == 0);
    std::cout << "  [PASS] test_llm_reset" << std::endl;
}

void test_llm_config_accessor() {
    LlmConfig config;
    config.max_tokens = 512;
    LlmProcessor processor(config);
    assert(processor.config().max_tokens == 512);
    std::cout << "  [PASS] test_llm_config_accessor" << std::endl;
}

void test_llm_generate_not_initialized() {
    LlmConfig config;
    LlmProcessor processor(config);
    auto token = processor.generate_next();
    assert(token.text.empty());
    std::cout << "  [PASS] test_llm_generate_not_initialized" << std::endl;
}

void test_llm_generate_no_messages() {
    LlmConfig config;
    LlmProcessor processor(config);
    processor.initialize();
    auto token = processor.generate_next();
    assert(token.text.empty());
    std::cout << "  [PASS] test_llm_generate_no_messages" << std::endl;
}

int main() {
    std::cout << "Running LLM tests..." << std::endl;
    test_llm_config_default();
    test_llm_processor_new();
    test_llm_initialize();
    test_llm_generate_next();
    test_llm_generate_multiple();
    test_llm_generate_complete();
    test_llm_reset();
    test_llm_config_accessor();
    test_llm_generate_not_initialized();
    test_llm_generate_no_messages();
    std::cout << "LLM tests: ALL PASSED" << std::endl;
    return 0;
}
