/// Session management header.
///
/// Manages inference engine session state.

#pragma once

#include <string>
#include <cstdint>
#include <atomic>

namespace szca {

/// Session state.
enum class SessionState {
    Created,
    Active,
    Paused,
    Ended
};

/// Session statistics.
struct SessionStats {
    uint64_t audio_in_bytes = 0;
    uint64_t audio_out_bytes = 0;
    uint32_t stt_partials = 0;
    uint32_t stt_finals = 0;
    uint32_t llm_tokens = 0;
    uint32_t tts_chunks = 0;
    double last_latency_ms = 0.0;
};

/// Session manager.
class Session {
public:
    Session();
    ~Session();

    // std::atomic members delete the implicit copy/move assignment, but
    // Engine::reset() needs to reset a Session by value. Provide explicit
    // assignment that copies the atomic's current value.
    Session(const Session& other);
    Session& operator=(const Session& other);

    /// Get session ID.
    const std::string& id() const;

    /// Get current state.
    SessionState state() const;

    /// Activate session.
    bool activate();

    /// Pause session.
    bool pause();

    /// Resume session.
    bool resume();

    /// End session.
    void end();

    /// Check if cancelled.
    bool is_cancelled() const;

    /// Cancel session.
    void cancel();

    /// Reset cancel flag.
    void reset_cancel();

    /// Get stats.
    const SessionStats& stats() const;

    /// Record stats.
    void record_bytes_in(uint64_t bytes);
    void record_bytes_out(uint64_t bytes);
    void record_stt_partial();
    void record_stt_final();
    void record_llm_token();
    void record_tts_chunk();
    void record_latency(double ms);

private:
    std::string id_;
    SessionState state_;
    SessionStats stats_;
    std::atomic<bool> cancelled_;
};

} // namespace szca
