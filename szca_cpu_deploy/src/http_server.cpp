/// Simple HTTP server implementation for SZCA CPU deployment.

#include "http_server.h"
#include <iostream>
#include <sstream>
#include <cstring>
#include <cstdlib>
#include <sys/socket.h>
#include <netinet/in.h>
#include <unistd.h>
#include <poll.h>
#include <errno.h>

namespace szca {

HttpServer::HttpServer(int port, Engine* engine)
    : port_(port)
    , engine_(engine)
    , running_(false)
    , server_fd_(-1) {
}

HttpServer::~HttpServer() {
    stop();
}

void HttpServer::start() {
    running_.store(true);

    // Spin up the fixed worker pool.
    workers_.reserve(kNumWorkers);
    for (size_t i = 0; i < kNumWorkers; i++) {
        workers_.emplace_back(&HttpServer::worker_loop, this);
    }

    server_thread_ = std::thread(&HttpServer::accept_connections, this);
}

void HttpServer::stop() {
    // Idempotent: only run shutdown once.
    if (!running_.exchange(false)) {
        // Still make sure any partially-started threads are joined.
    }

    // M13: nudge accept() awake by shutting down the listening socket. The
    // poll() timeout in the accept loop is what actually guarantees a bounded
    // shutdown on all platforms; this is a best-effort speedup.
    int fd = server_fd_.load();
    if (fd >= 0) {
        ::shutdown(fd, SHUT_RDWR);
    }

    if (server_thread_.joinable()) {
        server_thread_.join();
    }

    // Wake all workers so they can observe running_==false and exit.
    queue_cv_.notify_all();

    // H9: join every worker BEFORE members (engine_) are destroyed.
    for (auto& w : workers_) {
        if (w.joinable()) w.join();
    }
    workers_.clear();

    // Drain any client fds still queued so we don't leak them.
    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        while (!conn_queue_.empty()) {
            int fd = conn_queue_.front();
            conn_queue_.pop_front();
            if (fd >= 0) ::close(fd);
        }
    }

    // Close the listening socket exactly once.
    int close_fd = server_fd_.exchange(-1);
    if (close_fd >= 0) {
        if (::close(close_fd) < 0) {
            std::cerr << "[HTTP] Warning: close(server_fd) failed: "
                      << std::strerror(errno) << std::endl;
        }
    }
}

bool HttpServer::is_running() const {
    return running_.load();
}

void HttpServer::accept_connections() {
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) {
        std::cerr << "[HTTP] Failed to create socket" << std::endl;
        running_.store(false);
        queue_cv_.notify_all();
        return;
    }

    int opt = 1;
    if (setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt)) < 0) {
        std::cerr << "[HTTP] Warning: setsockopt(SO_REUSEADDR) failed: "
                  << std::strerror(errno) << std::endl;
    }

    struct sockaddr_in address;
    std::memset(&address, 0, sizeof(address));
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = INADDR_ANY;
    address.sin_port = htons(static_cast<uint16_t>(port_));

    if (bind(fd, (struct sockaddr*)&address, sizeof(address)) < 0) {
        std::cerr << "[HTTP] Failed to bind to port " << port_ << std::endl;
        ::close(fd);
        running_.store(false);
        queue_cv_.notify_all();
        return;
    }

    if (listen(fd, 10) < 0) {
        std::cerr << "[HTTP] Failed to listen" << std::endl;
        ::close(fd);
        running_.store(false);
        queue_cv_.notify_all();
        return;
    }

    // Publish the listening fd so stop() can shut it down.
    server_fd_.store(fd);

    while (running_.load()) {
        // M13: poll() the listening socket with a timeout so this loop wakes
        // periodically to re-check running_. accept() does NOT honor
        // SO_RCVTIMEO on macOS/BSD, and shutdown() does not reliably unblock a
        // blocked accept() there, so poll() is the portable way to guarantee
        // stop() never hangs.
        struct pollfd pfd;
        pfd.fd = fd;
        pfd.events = POLLIN;
        int pr = poll(&pfd, 1, 200 /* ms */);
        if (pr <= 0) {
            // pr == 0: timeout -> re-check running_. pr < 0: error/EINTR.
            if (pr < 0 && errno != EINTR) {
                if (!running_.load()) break;
            }
            continue;
        }

        struct sockaddr_in client_addr;
        socklen_t client_len = sizeof(client_addr);
        int client_fd = accept(fd, (struct sockaddr*)&client_addr, &client_len);

        if (client_fd < 0) {
            // accept() returns error when the socket is shut down in stop().
            if (!running_.load()) break;
            continue;
        }

        // Enqueue for a worker; drop (503-ish: just close) if the queue is full.
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            if (conn_queue_.size() >= kMaxQueued) {
                lock.unlock();
                std::cerr << "[HTTP] Connection queue full, dropping client" << std::endl;
                ::close(client_fd);
                continue;
            }
            conn_queue_.push_back(client_fd);
        }
        queue_cv_.notify_one();
    }
}

void HttpServer::worker_loop() {
    for (;;) {
        int client_fd = -1;
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            queue_cv_.wait(lock, [this] {
                return !running_.load() || !conn_queue_.empty();
            });
            if (!running_.load() && conn_queue_.empty()) {
                return;
            }
            client_fd = conn_queue_.front();
            conn_queue_.pop_front();
        }
        if (client_fd >= 0) {
            handle_client(client_fd);
        }
    }
}

void HttpServer::handle_client(int client_fd) {
    // L12: read until we have the full request (headers + Content-Length body),
    // capped at kMaxRequestBytes so a client cannot force unbounded growth.
    std::string request;
    char buffer[8192];
    size_t header_end = std::string::npos;
    size_t content_length = 0;
    bool have_content_length = false;

    for (;;) {
        ssize_t n = read(client_fd, buffer, sizeof(buffer));
        if (n < 0) {
            if (errno == EINTR) continue;
            break;
        }
        if (n == 0) break;  // peer closed

        request.append(buffer, static_cast<size_t>(n));
        if (request.size() > kMaxRequestBytes) {
            std::cerr << "[HTTP] Request exceeds max size, truncating" << std::endl;
            break;
        }

        // Once headers are complete, decide whether more body is needed.
        if (header_end == std::string::npos) {
            header_end = request.find("\r\n\r\n");
            if (header_end != std::string::npos) {
                // Case-insensitive-ish search for Content-Length.
                size_t cl = request.find("Content-Length:");
                if (cl == std::string::npos) cl = request.find("content-length:");
                if (cl != std::string::npos && cl < header_end) {
                    have_content_length = true;
                    content_length = static_cast<size_t>(
                        std::strtoul(request.c_str() + cl + 15, nullptr, 10));
                }
            }
        }

        if (header_end != std::string::npos) {
            if (!have_content_length) break;  // no body expected
            size_t body_have = request.size() - (header_end + 4);
            if (body_have >= content_length) break;  // full body received
        }
    }

    if (request.empty()) {
        ::close(client_fd);
        return;
    }

    // Parse method and path
    std::istringstream iss(request);
    std::string method, path, version;
    iss >> method >> path >> version;

    // Extract body
    size_t body_start = request.find("\r\n\r\n");
    std::string body = (body_start != std::string::npos) ? request.substr(body_start + 4) : "";

    // Handle request
    std::string response = handle_request(method, path, body);

    // L12: loop on write() to handle partial writes.
    const char* out = response.c_str();
    size_t remaining = response.size();
    while (remaining > 0) {
        ssize_t w = write(client_fd, out, remaining);
        if (w < 0) {
            if (errno == EINTR) continue;
            std::cerr << "[HTTP] write() failed: " << std::strerror(errno) << std::endl;
            break;
        }
        if (w == 0) break;
        out += w;
        remaining -= static_cast<size_t>(w);
    }

    if (::close(client_fd) < 0) {
        std::cerr << "[HTTP] Warning: close(client_fd) failed: "
                  << std::strerror(errno) << std::endl;
    }
}

std::string HttpServer::handle_request(const std::string& method, const std::string& path, const std::string& body) {
    std::string content_type = "Content-Type: application/json\r\n";

    if (path == "/health" && method == "GET") {
        return "HTTP/1.1 200 OK\r\n" + content_type + "\r\n" + handle_health();
    }

    if (path == "/v1/stt/stream" && method == "POST") {
        return "HTTP/1.1 200 OK\r\n" + content_type + "\r\n" + handle_stt(body);
    }

    if (path == "/v1/llm/stream" && method == "POST") {
        return "HTTP/1.1 200 OK\r\n" + content_type + "\r\n" + handle_llm(body);
    }

    if (path == "/v1/tts/stream" && method == "POST") {
        return "HTTP/1.1 200 OK\r\n" + content_type + "\r\n" + handle_tts(body);
    }

    return "HTTP/1.1 404 Not Found\r\nContent-Type: application/json\r\n\r\n{\"error\":\"Not found\"}";
}

std::string HttpServer::handle_health() {
    return "{\"status\":\"ok\",\"engine\":\"szca-cpu\",\"version\":\"5.0.0\"}";
}

std::string HttpServer::handle_stt(const std::string& body) {
    // Stub: In production, process audio and return transcription
    return "{\"type\":\"partial\",\"text\":\"STT processing...\",\"confidence\":0.9}";
}

std::string HttpServer::handle_llm(const std::string& body) {
    // Stub: In production, generate response tokens
    return "{\"type\":\"token\",\"text\":\"I'm doing great!\",\"token_id\":1}";
}

std::string HttpServer::handle_tts(const std::string& body) {
    // Stub: In production, generate audio chunks
    return "{\"type\":\"audio_chunk\",\"sample_rate\":16000,\"duration_ms\":20}";
}

} // namespace szca
