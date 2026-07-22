/// Session management implementation.

#include "session.h"
#include <random>
#include <sstream>

namespace szca {

static std::string generate_id() {
    // thread_local so each thread has its own generator: a function-local
    // `static std::mt19937` is not thread-safe (data race on shared state).
    // NOTE: these IDs are for correlation only, NOT cryptographically secure.
    thread_local std::random_device rd;
    thread_local std::mt19937 gen(rd());
    thread_local std::uniform_int_distribution<> dis(0, 15);

    std::stringstream ss;
    for (int i = 0; i < 16; i++) {
        ss << std::hex << dis(gen);
    }
    return ss.str();
}

Session::Session()
    : id_(generate_id())
    , state_(SessionState::Created)
    , stats_{}
    , cancelled_(false) {
}

Session::~Session() = default;

Session::Session(const Session& other)
    : id_(other.id_)
    , state_(other.state_)
    , stats_(other.stats_)
    , cancelled_(other.cancelled_.load()) {
}

Session& Session::operator=(const Session& other) {
    if (this != &other) {
        id_ = other.id_;
        state_ = other.state_;
        stats_ = other.stats_;
        cancelled_.store(other.cancelled_.load());
    }
    return *this;
}

const std::string& Session::id() const {
    return id_;
}

SessionState Session::state() const {
    return state_;
}

bool Session::activate() {
    if (state_ != SessionState::Created) return false;
    state_ = SessionState::Active;
    return true;
}

bool Session::pause() {
    if (state_ != SessionState::Active) return false;
    state_ = SessionState::Paused;
    return true;
}

bool Session::resume() {
    if (state_ != SessionState::Paused) return false;
    state_ = SessionState::Active;
    return true;
}

void Session::end() {
    state_ = SessionState::Ended;
    cancelled_.store(true);
}

bool Session::is_cancelled() const {
    return cancelled_.load();
}

void Session::cancel() {
    cancelled_.store(true);
}

void Session::reset_cancel() {
    cancelled_.store(false);
}

const SessionStats& Session::stats() const {
    return stats_;
}

void Session::record_bytes_in(uint64_t bytes) {
    stats_.audio_in_bytes += bytes;
}

void Session::record_bytes_out(uint64_t bytes) {
    stats_.audio_out_bytes += bytes;
}

void Session::record_stt_partial() {
    stats_.stt_partials++;
}

void Session::record_stt_final() {
    stats_.stt_finals++;
}

void Session::record_llm_token() {
    stats_.llm_tokens++;
}

void Session::record_tts_chunk() {
    stats_.tts_chunks++;
}

void Session::record_latency(double ms) {
    stats_.last_latency_ms = ms;
}

} // namespace szca
