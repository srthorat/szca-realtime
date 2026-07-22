/// IPC module implementation.
///
/// POSIX shared memory interface for gateway ↔ engine communication.

#include "ipc.h"
#include <cstring>
#include <algorithm>

namespace szca {

IpcChannel::IpcChannel(const IpcConfig& config)
    : config_(config)
    , connected_(false)
    , write_pos_(0)
    , read_pos_(0)
    , buffer_() {
    // Guard against divide-by-zero in the ring-buffer modulo arithmetic.
    if (config_.buffer_capacity <= 0) {
        config_.buffer_capacity = 1;
    }
    if (config_.chunk_size < 0) {
        config_.chunk_size = 0;
    }
    buffer_.resize(config_.buffer_capacity);
    lengths_.assign(config_.buffer_capacity, 0);
    for (auto& slot : buffer_) {
        slot.resize(config_.chunk_size, 0);
    }
}

IpcChannel::~IpcChannel() = default;

bool IpcChannel::initialize() {
    // In production: create/open POSIX SHM via shm_open
    connected_ = true;
    return true;
}

bool IpcChannel::write(const uint8_t* data, int size) {
    if (!connected_) return false;
    if (!data || size < 0 || size > config_.chunk_size) return false;

    int next_pos = (write_pos_.load() + 1) % config_.buffer_capacity;
    if (next_pos == read_pos_.load()) {
        return false; // Buffer full
    }

    int wp = write_pos_.load();
    std::memcpy(buffer_[wp].data(), data, size);
    lengths_[wp] = size;  // remember how many bytes are actually valid
    write_pos_.store(next_pos);
    return true;
}

int IpcChannel::read(uint8_t* buffer, int max_size) {
    if (!connected_) return 0;
    if (!buffer || max_size <= 0) return 0;
    if (is_empty()) return 0;

    int rp = read_pos_.load();
    // Only return the bytes that were actually written into this slot,
    // never the stale tail of the fixed-size chunk buffer.
    int bytes_to_read = std::min(max_size, lengths_[rp]);
    if (bytes_to_read < 0) bytes_to_read = 0;
    std::memcpy(buffer, buffer_[rp].data(), bytes_to_read);
    read_pos_.store((rp + 1) % config_.buffer_capacity);

    return bytes_to_read;
}

bool IpcChannel::is_connected() const {
    return connected_;
}

int IpcChannel::len() const {
    return (write_pos_.load() - read_pos_.load() + config_.buffer_capacity) % config_.buffer_capacity;
}

bool IpcChannel::is_empty() const {
    return write_pos_.load() == read_pos_.load();
}

bool IpcChannel::is_full() const {
    return (write_pos_.load() + 1) % config_.buffer_capacity == read_pos_.load();
}

const IpcConfig& IpcChannel::config() const {
    return config_;
}

} // namespace szca
