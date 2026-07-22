/// Session module unit tests.

#include "session.h"
#include <cassert>
#include <iostream>

using namespace szca;

void test_session_new() {
    Session session;
    assert(session.state() == SessionState::Created);
    assert(!session.is_cancelled());
    assert(!session.id().empty());
    std::cout << "  [PASS] test_session_new" << std::endl;
}

void test_session_unique_id() {
    Session s1;
    Session s2;
    assert(s1.id() != s2.id());
    std::cout << "  [PASS] test_session_unique_id" << std::endl;
}

void test_session_activate() {
    Session session;
    assert(session.activate());
    assert(session.state() == SessionState::Active);
    std::cout << "  [PASS] test_session_activate" << std::endl;
}

void test_session_activate_invalid() {
    Session session;
    session.activate();
    assert(!session.activate()); // Already active
    std::cout << "  [PASS] test_session_activate_invalid" << std::endl;
}

void test_session_pause() {
    Session session;
    session.activate();
    assert(session.pause());
    assert(session.state() == SessionState::Paused);
    std::cout << "  [PASS] test_session_pause" << std::endl;
}

void test_session_pause_invalid() {
    Session session;
    assert(!session.pause()); // Not active
    std::cout << "  [PASS] test_session_pause_invalid" << std::endl;
}

void test_session_resume() {
    Session session;
    session.activate();
    session.pause();
    assert(session.resume());
    assert(session.state() == SessionState::Active);
    std::cout << "  [PASS] test_session_resume" << std::endl;
}

void test_session_resume_invalid() {
    Session session;
    session.activate();
    assert(!session.resume()); // Not paused
    std::cout << "  [PASS] test_session_resume_invalid" << std::endl;
}

void test_session_end() {
    Session session;
    session.end();
    assert(session.state() == SessionState::Ended);
    assert(session.is_cancelled());
    std::cout << "  [PASS] test_session_end" << std::endl;
}

void test_session_cancel() {
    Session session;
    session.cancel();
    assert(session.is_cancelled());
    std::cout << "  [PASS] test_session_cancel" << std::endl;
}

void test_session_reset_cancel() {
    Session session;
    session.cancel();
    session.reset_cancel();
    assert(!session.is_cancelled());
    std::cout << "  [PASS] test_session_reset_cancel" << std::endl;
}

void test_session_stats() {
    Session session;
    session.record_bytes_in(1000);
    session.record_bytes_out(2000);
    session.record_stt_partial();
    session.record_stt_partial();
    session.record_stt_final();
    session.record_llm_token();
    session.record_tts_chunk();
    session.record_latency(42.5);

    auto& stats = session.stats();
    assert(stats.audio_in_bytes == 1000);
    assert(stats.audio_out_bytes == 2000);
    assert(stats.stt_partials == 2);
    assert(stats.stt_finals == 1);
    assert(stats.llm_tokens == 1);
    assert(stats.tts_chunks == 1);
    assert(stats.last_latency_ms == 42.5);
    std::cout << "  [PASS] test_session_stats" << std::endl;
}

int main() {
    std::cout << "Running Session tests..." << std::endl;
    test_session_new();
    test_session_unique_id();
    test_session_activate();
    test_session_activate_invalid();
    test_session_pause();
    test_session_pause_invalid();
    test_session_resume();
    test_session_resume_invalid();
    test_session_end();
    test_session_cancel();
    test_session_reset_cancel();
    test_session_stats();
    std::cout << "Session tests: ALL PASSED" << std::endl;
    return 0;
}
