/// ONNX Runtime utility wrapper.
///
/// Provides a safe C++ wrapper around ONNX Runtime C API.
/// Handles session creation, input/output binding, and inference execution.

#pragma once

#include <string>
#include <vector>
#include <cstdint>
#include <memory>
#include <functional>

// Forward declare ORT types to avoid header dependency
struct OrtEnv;
struct OrtSession;
struct OrtSessionOptions;
struct OrtMemoryInfo;
struct OrtValue;
struct OrtAllocator;

namespace szca {

/// ORT inference session wrapper.
class OrtSessionWrapper {
public:
    OrtSessionWrapper();
    ~OrtSessionWrapper();

    OrtSessionWrapper(const OrtSessionWrapper&) = delete;
    OrtSessionWrapper& operator=(const OrtSessionWrapper&) = delete;

    /// Load an ONNX model from disk.
    bool load(const std::string& model_path, int device_id = 0);

    /// Run inference with float input tensors.
    /// input_names: names of input nodes
    /// input_shapes: shapes of input tensors
    /// input_data: flat float arrays for each input
    /// Returns output tensors as flat float vectors.
    std::vector<std::vector<float>> run(
        const std::vector<std::string>& input_names,
        const std::vector<std::vector<int64_t>>& input_shapes,
        const std::vector<std::vector<float>>& input_data
    );

    /// Run inference with int16 input tensors (for audio).
    std::vector<std::vector<float>> run_int16(
        const std::vector<std::string>& input_names,
        const std::vector<std::vector<int64_t>>& input_shapes,
        const std::vector<std::vector<int16_t>>& input_data
    );

    /// Check if session is loaded.
    bool is_loaded() const;

    /// Get input count.
    size_t input_count() const;

    /// Get output count.
    size_t output_count() const;

    /// Get input name by index.
    std::string input_name(size_t index) const;

    /// Get output name by index.
    std::string output_name(size_t index) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

/// ORT Environment singleton.
class OrtEnvironment {
public:
    static OrtEnvironment& instance();

    /// Get the ORT env pointer.
    OrtEnv* env() const;

private:
    OrtEnvironment();
    ~OrtEnvironment();
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace szca
