/// Unified inference engine implementation.
///
/// Orchestrates STT → LLM → TTS pipeline.

#include "engine.h"

namespace szca {

Engine::Engine(const EngineConfig& config)
    : config_(config)
    , initialized_(false)
    , session_()
    , stt_(std::make_unique<SttProcessor>(config.stt))
    , llm_(std::make_unique<LlmProcessor>(config.llm))
    , tts_(std::make_unique<TtsProcessor>(config.tts))
    , resampler_(std::make_unique<Resampler>(config.resampler)) {
}

Engine::~Engine() = default;

bool Engine::initialize() {
    if (!stt_->initialize()) return false;
    if (!llm_->initialize()) return false;
    if (!tts_->initialize()) return false;
    if (!resampler_->initialize()) return false;

    initialized_ = true;
    return true;
}

void Engine::process_audio(
    const int16_t* pcm_input,
    int input_samples,
    AudioChunkCallback audio_callback,
    TextTokenCallback text_callback
) {
    // L16: bound the input to avoid unbounded allocation / overflow in the
    // downstream byte-count math and STT float conversion.
    static constexpr int MAX_SAMPLES = 16 * 1024 * 1024;

    if (!initialized_ || !pcm_input || input_samples <= 0 ||
        input_samples > MAX_SAMPLES) {
        return;
    }

    session_.record_bytes_in(static_cast<uint64_t>(input_samples) * sizeof(int16_t));

    // Step 1: STT — speech to text
    auto stt_result = stt_->process(pcm_input, input_samples);
    if (stt_result) {
        if (stt_result->type == SttResult::Partial) {
            session_.record_stt_partial();
            if (text_callback) {
                text_callback(stt_result->text, false);
            }
        } else {
            session_.record_stt_final();
            if (text_callback) {
                text_callback(stt_result->text, true);
            }

            // Step 2: LLM — generate response
            llm_->add_message("user", stt_result->text);

            while (true) {
                auto token = llm_->generate_next();
                if (token.text.empty()) break;

                session_.record_llm_token();
                if (text_callback) {
                    text_callback(token.text, false);
                }

                // Step 3: TTS — text to speech (per sentence)
                if (token.text.find('.') != std::string::npos ||
                    token.text.find('?') != std::string::npos ||
                    token.text.find('!') != std::string::npos) {

                    auto tts_chunks = tts_->synthesize(token.text);
                    for (auto& chunk : tts_chunks) {
                        session_.record_tts_chunk();

                        // Step 4: Resample 24kHz → 16kHz
                        auto resampled = resampler_->process(
                            chunk.pcm.data(),
                            static_cast<int>(chunk.pcm.size())
                        );

                        session_.record_bytes_out(resampled.size() * sizeof(int16_t));
                        if (audio_callback) {
                            audio_callback(
                                resampled.data(),
                                static_cast<int>(resampled.size()),
                                config_.resampler.output_sample_rate
                            );
                        }
                    }
                }
            }
        }
    }
}

void Engine::reset() {
    if (stt_) stt_ = std::make_unique<SttProcessor>(config_.stt);
    if (llm_) llm_->reset();
    session_ = Session();
}

bool Engine::is_initialized() const {
    return initialized_;
}

Session& Engine::session() {
    return session_;
}

const EngineConfig& Engine::config() const {
    return config_;
}

} // namespace szca
