/// Unified inference engine header.
///
/// Orchestrates STT → LLM → TTS pipeline.

#pragma once

#include "stt.h"
#include "llm.h"
#include "tts.h"
#include "resampler.h"
#include "session.h"
#include <functional>
#include <memory>

namespace szca {

/// Engine configuration.
struct EngineConfig {
    SttConfig stt;
    LlmConfig llm;
    TtsConfig tts;
    ResamplerConfig resampler;
};

/// Engine callback for streaming output.
using AudioChunkCallback = std::function<void(const int16_t* pcm, int samples, int sample_rate)>;
using TextTokenCallback = std::function<void(const std::string& token, bool is_final)>;

/// Unified inference engine.
class Engine {
public:
    explicit Engine(const EngineConfig& config);
    ~Engine();

    Engine(const Engine&) = delete;
    Engine& operator=(const Engine&) = delete;

    /// Initialize all components.
    bool initialize();

    /// Process audio input through the full pipeline.
    /// Calls callbacks for streaming output.
    void process_audio(
        const int16_t* pcm_input,
        int input_samples,
        AudioChunkCallback audio_callback,
        TextTokenCallback text_callback
    );

    /// Reset engine state.
    void reset();

    /// Check if initialized.
    bool is_initialized() const;

    /// Get session.
    Session& session();

    /// Get configuration.
    const EngineConfig& config() const;

private:
    EngineConfig config_;
    bool initialized_;
    Session session_;
    std::unique_ptr<SttProcessor> stt_;
    std::unique_ptr<LlmProcessor> llm_;
    std::unique_ptr<TtsProcessor> tts_;
    std::unique_ptr<Resampler> resampler_;
};

} // namespace szca
