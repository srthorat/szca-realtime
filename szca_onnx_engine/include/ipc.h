/// IPC (Inter-Process Communication) module header.
///
/// POSIX shared memory interface for gateway ↔ engine communication.
/// Target latency: <0.1ms per operation

#pragma once

#include <string>
#include <vector>
#include <cstdint>
#include <atomic>

namespace szca {

/// IPC configuration.
struct IpcConfig {
    std::string shm_prefix = "/dev/shm/szca";
    int buffer_capacity = 256;
    int chunk_size = 640;  // 20ms @ 16kHz 16-bit mono
};

/// IPC channel for bidirectional communication.
class IpcChannel {
public:
    explicit IpcChannel(const IpcConfig& config);
    ~IpcChannel();

    IpcChannel(const IpcChannel&) = delete;
    IpcChannel& operator=(const IpcChannel&) = delete;

    /// Initialize the shared memory segment.
    bool initialize();

    /// Write audio data to shared memory.
    /// Returns true if successful, false if buffer full.
    bool write(const uint8_t* data, int size);

    /// Read audio data from shared memory.
    /// Returns number of bytes read, 0 if empty.
    int read(uint8_t* buffer, int max_size);

    /// Check if channel is connected.
    bool is_connected() const;

    /// Get number of items in buffer.
    int len() const;

    /// Check if buffer is empty.
    bool is_empty() const;

    /// Check if buffer is full.
    bool is_full() const;

    /// Get configuration.
    const IpcConfig& config() const;

private:
    IpcConfig config_;
    bool connected_;
    std::atomic<int> write_pos_;
    std::atomic<int> read_pos_;
    std::vector<std::vector<uint8_t>> buffer_;
    std::vector<int> lengths_;  // per-slot count of valid bytes written
};

} // namespace szca
