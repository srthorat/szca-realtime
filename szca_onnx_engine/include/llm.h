/// LLM (Language Model) module header.
///
/// Hermes-3-Llama-3.2-3B INT8 streaming text generation.
/// License: Apache 2.0 (Meta AI / Nous Research)

#pragma once

#include <string>
#include <vector>
#include <cstdint>
#include <functional>
#include <memory>

namespace szca {

/// LLM configuration.
struct LlmConfig {
    std::string model_path = "./models/hermes-3-3b-int8.onnx";
    int max_tokens = 256;
    float temperature = 0.7f;
    float top_p = 0.9f;
    int context_length = 4096;
};

/// LLM token result.
struct LlmToken {
    std::string text;
    int token_id;
    float logprob;
    int index;
};

/// LLM completion result.
struct LlmCompletion {
    std::string full_text;
    int total_tokens;
    std::string finish_reason;
};

/// LLM processor interface.
class LlmProcessor {
public:
    explicit LlmProcessor(const LlmConfig& config);
    ~LlmProcessor();

    LlmProcessor(const LlmProcessor&) = delete;
    LlmProcessor& operator=(const LlmProcessor&) = delete;

    /// Initialize the processor (load model).
    bool initialize();

    /// Add a message to the conversation context.
    void add_message(const std::string& role, const std::string& content);

    /// Generate the next token.
    /// Returns the token, or empty if generation is complete.
    LlmToken generate_next();

    /// Generate all tokens until completion.
    LlmCompletion generate_complete();

    /// Reset conversation context.
    void reset();

    /// Check if processor is initialized.
    bool is_initialized() const;

    /// Get configuration.
    const LlmConfig& config() const;

    /// Get total tokens generated.
    uint64_t token_count() const;

private:
    LlmConfig config_;
    bool initialized_;
    uint64_t token_count_;
    std::vector<std::pair<std::string, std::string>> messages_;
    std::string generated_text_;
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace szca
