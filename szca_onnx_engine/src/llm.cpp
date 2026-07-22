/// LLM (Language Model) module implementation.
///
/// Hermes-3-Llama-3.2-3B INT8 streaming text generation.

#include "llm.h"
#include "ort_utils.h"
#include <sstream>
#include <iostream>

namespace szca {

struct LlmProcessor::Impl {
    OrtSessionWrapper session;
    std::vector<std::pair<std::string, std::string>> messages;
    std::string generated_text;
    int token_index = 0;
};

LlmProcessor::LlmProcessor(const LlmConfig& config)
    : config_(config)
    , initialized_(false)
    , token_count_(0)
    , messages_()
    , generated_text_()
    , impl_(std::make_unique<Impl>()) {
}

LlmProcessor::~LlmProcessor() = default;

bool LlmProcessor::initialize() {
    if (!impl_->session.load(config_.model_path)) {
        std::cerr << "[LLM] Failed to load model: " << config_.model_path << std::endl;
        return false;
    }
    initialized_ = true;
    std::cout << "[LLM] Initialized: " << config_.model_path << std::endl;
    return true;
}

void LlmProcessor::add_message(const std::string& role, const std::string& content) {
    messages_.emplace_back(role, content);
    impl_->messages.emplace_back(role, content);
}

LlmToken LlmProcessor::generate_next() {
    if (!initialized_ || messages_.empty()) {
        return {"", -1, 0.0f, -1};
    }

    // In production: tokenize input, run ONNX GenAI, decode output
    // For now, simulate token generation with realistic delays
    static const std::vector<std::string> responses = {
        "I'm", " doing", " great,", " thanks!", " How", " can", " I", " help", " you", "?"
    };

    int idx = impl_->token_index % responses.size();
    impl_->token_index++;
    token_count_++;

    LlmToken token;
    token.text = responses[idx];
    token.token_id = static_cast<int>(token_count_);
    token.logprob = -0.05f;
    token.index = impl_->token_index - 1;
    generated_text_ += token.text;
    impl_->generated_text += token.text;

    return token;
}

LlmCompletion LlmProcessor::generate_complete() {
    LlmCompletion completion;
    std::string full_text;

    while (true) {
        auto token = generate_next();
        if (token.text.empty()) break;
        full_text += token.text;

        if (token.text.find('?') != std::string::npos ||
            token.text.find('!') != std::string::npos) {
            break;
        }

        if (static_cast<int>(token_count_) >= config_.max_tokens) {
            break;
        }
    }

    completion.full_text = full_text;
    completion.total_tokens = static_cast<int>(token_count_);
    completion.finish_reason = "stop";

    return completion;
}

void LlmProcessor::reset() {
    messages_.clear();
    impl_->messages.clear();
    generated_text_.clear();
    impl_->generated_text.clear();
    impl_->token_index = 0;
    token_count_ = 0;
}

bool LlmProcessor::is_initialized() const {
    return initialized_;
}

const LlmConfig& LlmProcessor::config() const {
    return config_;
}

uint64_t LlmProcessor::token_count() const {
    return token_count_;
}

} // namespace szca
