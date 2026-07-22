/// IPC module unit tests.

#include "ipc.h"
#include <cassert>
#include <iostream>
#include <cstring>

using namespace szca;

void test_ipc_config_default() {
    IpcConfig config;
    assert(config.shm_prefix.find("szca") != std::string::npos);
    assert(config.buffer_capacity == 256);
    assert(config.chunk_size == 640);
    std::cout << "  [PASS] test_ipc_config_default" << std::endl;
}

void test_ipc_channel_new() {
    IpcConfig config;
    IpcChannel channel(config);
    assert(!channel.is_connected());
    assert(channel.is_empty());
    assert(!channel.is_full());
    assert(channel.len() == 0);
    std::cout << "  [PASS] test_ipc_channel_new" << std::endl;
}

void test_ipc_initialize() {
    IpcConfig config;
    IpcChannel channel(config);
    assert(channel.initialize());
    assert(channel.is_connected());
    std::cout << "  [PASS] test_ipc_initialize" << std::endl;
}

void test_ipc_write_not_connected() {
    IpcConfig config;
    IpcChannel channel(config);
    uint8_t data[640] = {0};
    assert(!channel.write(data, 640));
    std::cout << "  [PASS] test_ipc_write_not_connected" << std::endl;
}

void test_ipc_read_not_connected() {
    IpcConfig config;
    IpcChannel channel(config);
    uint8_t buffer[640];
    assert(channel.read(buffer, 640) == 0);
    std::cout << "  [PASS] test_ipc_read_not_connected" << std::endl;
}

void test_ipc_write_read_single() {
    IpcConfig config;
    IpcChannel channel(config);
    channel.initialize();

    uint8_t data[4] = {1, 2, 3, 4};
    assert(channel.write(data, 4));
    assert(channel.len() == 1);

    uint8_t buffer[640];
    int bytes = channel.read(buffer, 640);
    assert(bytes == 4);
    assert(buffer[0] == 1 && buffer[1] == 2 && buffer[2] == 3 && buffer[3] == 4);
    std::cout << "  [PASS] test_ipc_write_read_single" << std::endl;
}

void test_ipc_write_read_fifo() {
    IpcConfig config;
    IpcChannel channel(config);
    channel.initialize();

    for (int i = 0; i < 10; i++) {
        uint8_t data[640];
        std::memset(data, i, 640);
        assert(channel.write(data, 640));
    }

    for (int i = 0; i < 10; i++) {
        uint8_t buffer[640];
        int bytes = channel.read(buffer, 640);
        assert(bytes == 640);
        assert(buffer[0] == i);
    }
    std::cout << "  [PASS] test_ipc_write_read_fifo" << std::endl;
}

void test_ipc_buffer_full() {
    IpcConfig config;
    config.buffer_capacity = 4;
    IpcChannel channel(config);
    channel.initialize();

    uint8_t data[640] = {0};
    assert(channel.write(data, 640));
    assert(channel.write(data, 640));
    assert(channel.write(data, 640));
    assert(!channel.write(data, 640)); // Full
    assert(channel.is_full());
    std::cout << "  [PASS] test_ipc_buffer_full" << std::endl;
}

void test_ipc_buffer_empty() {
    IpcConfig config;
    IpcChannel channel(config);
    channel.initialize();

    uint8_t buffer[640];
    assert(channel.read(buffer, 640) == 0);
    assert(channel.is_empty());
    std::cout << "  [PASS] test_ipc_buffer_empty" << std::endl;
}

void test_ipc_write_too_large() {
    IpcConfig config;
    IpcChannel channel(config);
    channel.initialize();

    uint8_t data[1000];
    assert(!channel.write(data, 1000));
    std::cout << "  [PASS] test_ipc_write_too_large" << std::endl;
}

void test_ipc_wraparound() {
    IpcConfig config;
    config.buffer_capacity = 4;
    IpcChannel channel(config);
    channel.initialize();

    uint8_t data[640] = {0};
    for (int round = 0; round < 10; round++) {
        for (int i = 0; i < 3; i++) {
            data[0] = static_cast<uint8_t>(round * 3 + i);
            assert(channel.write(data, 640));
        }
        for (int i = 0; i < 3; i++) {
            uint8_t buffer[640];
            channel.read(buffer, 640);
            assert(buffer[0] == static_cast<uint8_t>(round * 3 + i));
        }
    }
    std::cout << "  [PASS] test_ipc_wraparound" << std::endl;
}

void test_ipc_config_accessor() {
    IpcConfig config;
    config.chunk_size = 1280;
    IpcChannel channel(config);
    assert(channel.config().chunk_size == 1280);
    std::cout << "  [PASS] test_ipc_config_accessor" << std::endl;
}

int main() {
    std::cout << "Running IPC tests..." << std::endl;
    test_ipc_config_default();
    test_ipc_channel_new();
    test_ipc_initialize();
    test_ipc_write_not_connected();
    test_ipc_read_not_connected();
    test_ipc_write_read_single();
    test_ipc_write_read_fifo();
    test_ipc_buffer_full();
    test_ipc_buffer_empty();
    test_ipc_write_too_large();
    test_ipc_wraparound();
    test_ipc_config_accessor();
    std::cout << "IPC tests: ALL PASSED" << std::endl;
    return 0;
}
