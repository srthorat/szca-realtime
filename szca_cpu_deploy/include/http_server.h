/// Simple HTTP server for SZCA CPU deployment.
///
/// Handles SSE streaming for STT, LLM, TTS endpoints.

#pragma once

#include "engine.h"
#include <string>
#include <thread>
#include <vector>
#include <deque>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <functional>

namespace szca {

/// Simple HTTP server using raw sockets.
class HttpServer {
public:
    HttpServer(int port, Engine* engine);
    ~HttpServer();

    /// Start the server (non-blocking).
    void start();

    /// Stop the server.
    void stop();

    /// Check if server is running.
    bool is_running() const;

private:
    int port_;
    Engine* engine_;
    std::atomic<bool> running_;
    std::thread server_thread_;
    std::atomic<int> server_fd_;  // listening socket; -1 when not open

    // Fixed-size worker pool with a bounded connection queue. Workers are
    // joined in stop()/destructor BEFORE engine_/this are destroyed, so no
    // detached thread can dereference freed memory (H9). The queue is capped
    // so we never spawn unbounded work.
    static constexpr size_t kNumWorkers = 8;
    static constexpr size_t kMaxQueued = 128;
    static constexpr size_t kMaxRequestBytes = 1 << 20;  // 1 MiB request cap

    std::vector<std::thread> workers_;
    std::deque<int> conn_queue_;      // pending client fds
    std::mutex queue_mutex_;
    std::condition_variable queue_cv_;

    void accept_connections();
    void worker_loop();
    void handle_client(int client_fd);
    std::string handle_request(const std::string& method, const std::string& path, const std::string& body);
    std::string handle_stt(const std::string& body);
    std::string handle_llm(const std::string& body);
    std::string handle_tts(const std::string& body);
    std::string handle_health();
};

} // namespace szca
