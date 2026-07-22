/// ONNX Runtime utility implementation.
///
/// The ONNX Runtime C API is a versioned vtable: all functions are reached
/// through the `OrtApi*` returned by `OrtGetApiBase()->GetApi(ORT_API_VERSION)`
/// (there are no `OrtCreateEnv`-style free functions). `ort_api()` caches that
/// pointer. CUDA execution-provider selection is intentionally omitted here:
/// the Homebrew/CPU build ships no CUDA provider. To enable GPU, append the
/// provider in `OrtSessionWrapper()` via
/// `OrtSessionOptionsAppendExecutionProvider_CUDA_V2(...)` guarded by a
/// GPU-build macro, and pass `device_id` through from `load()`.

#include "ort_utils.h"

#ifdef HAS_ONNXRUNTIME
#include <onnxruntime_c_api.h>
#endif

#include <iostream>
#include <stdexcept>

namespace szca {

#ifdef HAS_ONNXRUNTIME
namespace {
/// Cached ONNX Runtime API vtable for this ABI version.
const OrtApi* ort_api() {
    static const OrtApi* api = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    return api;
}

/// Log an ORT status, release it, and return true if it represented an error.
bool ort_failed(OrtStatus* status, const char* context) {
    if (!status) return false;
    const OrtApi* g = ort_api();
    std::cerr << "[ORT] " << context << ": " << g->GetErrorMessage(status) << std::endl;
    g->ReleaseStatus(status);
    return true;
}
} // namespace
#endif

// ============================================================================
// OrtEnvironment implementation
// ============================================================================

struct OrtEnvironment::Impl {
#ifdef HAS_ONNXRUNTIME
    OrtEnv* env = nullptr;
#endif
};

OrtEnvironment::OrtEnvironment() : impl_(std::make_unique<Impl>()) {
#ifdef HAS_ONNXRUNTIME
    OrtStatus* status = ort_api()->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "SZCA", &impl_->env);
    if (ort_failed(status, "CreateEnv failed")) {
        impl_->env = nullptr;
    }
#endif
}

OrtEnvironment::~OrtEnvironment() {
#ifdef HAS_ONNXRUNTIME
    if (impl_->env) {
        ort_api()->ReleaseEnv(impl_->env);
    }
#endif
}

OrtEnvironment& OrtEnvironment::instance() {
    static OrtEnvironment env;
    return env;
}

OrtEnv* OrtEnvironment::env() const {
#ifdef HAS_ONNXRUNTIME
    return impl_->env;
#else
    return nullptr;
#endif
}

// ============================================================================
// OrtSessionWrapper implementation
// ============================================================================

struct OrtSessionWrapper::Impl {
#ifdef HAS_ONNXRUNTIME
    OrtSession* session = nullptr;
    OrtSessionOptions* options = nullptr;
    OrtAllocator* allocator = nullptr;  // owned by ORT; must NOT be released
#endif
    bool loaded = false;
    std::vector<std::string> input_names_;
    std::vector<std::string> output_names_;
};

OrtSessionWrapper::OrtSessionWrapper() : impl_(std::make_unique<Impl>()) {
#ifdef HAS_ONNXRUNTIME
    const OrtApi* g = ort_api();

    OrtStatus* status = g->CreateSessionOptions(&impl_->options);
    if (ort_failed(status, "CreateSessionOptions failed")) {
        impl_->options = nullptr;
        return;
    }

    // NOTE: CUDA execution provider is not appended in the CPU build (see the
    // file header). The default provider is CPU.

    // GetAllocatorWithDefaultOptions returns a process-wide allocator that is
    // owned by ORT and must not be released.
    status = g->GetAllocatorWithDefaultOptions(&impl_->allocator);
    if (ort_failed(status, "GetAllocatorWithDefaultOptions failed")) {
        impl_->allocator = nullptr;
    }
#endif
}

OrtSessionWrapper::~OrtSessionWrapper() {
#ifdef HAS_ONNXRUNTIME
    const OrtApi* g = ort_api();
    if (impl_->session) g->ReleaseSession(impl_->session);
    if (impl_->options) g->ReleaseSessionOptions(impl_->options);
    // impl_->allocator is owned by ORT; do NOT release it.
#endif
}

bool OrtSessionWrapper::load(const std::string& model_path, int device_id) {
#ifdef HAS_ONNXRUNTIME
    const OrtApi* g = ort_api();
    if (!impl_->options || !impl_->allocator) return false;
    if (!OrtEnvironment::instance().env()) {
        std::cerr << "[ORT] Environment not initialized" << std::endl;
        return false;
    }

    // device_id would select the CUDA device; unused in the CPU build. Appending
    // the CUDA EP must happen on session options BEFORE CreateSession (see header).
    (void)device_id;

    // On POSIX, ORTCHAR_T == char, so c_str() is the correct model path type.
    OrtStatus* status = g->CreateSession(
        OrtEnvironment::instance().env(),
        model_path.c_str(),
        impl_->options,
        &impl_->session
    );
    if (ort_failed(status, ("CreateSession failed for " + model_path).c_str())) {
        impl_->session = nullptr;
        return false;
    }

    // Cache input names.
    size_t num_inputs = 0;
    status = g->SessionGetInputCount(impl_->session, &num_inputs);
    if (ort_failed(status, "SessionGetInputCount failed")) return false;
    impl_->input_names_.resize(num_inputs);
    for (size_t i = 0; i < num_inputs; i++) {
        char* name = nullptr;
        status = g->SessionGetInputName(impl_->session, i, impl_->allocator, &name);
        if (ort_failed(status, "SessionGetInputName failed")) return false;
        impl_->input_names_[i] = name ? name : "";
        if (name) ort_failed(g->AllocatorFree(impl_->allocator, name), "AllocatorFree(input name) failed");
    }

    // Cache output names.
    size_t num_outputs = 0;
    status = g->SessionGetOutputCount(impl_->session, &num_outputs);
    if (ort_failed(status, "SessionGetOutputCount failed")) return false;
    impl_->output_names_.resize(num_outputs);
    for (size_t i = 0; i < num_outputs; i++) {
        char* name = nullptr;
        status = g->SessionGetOutputName(impl_->session, i, impl_->allocator, &name);
        if (ort_failed(status, "SessionGetOutputName failed")) return false;
        impl_->output_names_[i] = name ? name : "";
        if (name) ort_failed(g->AllocatorFree(impl_->allocator, name), "AllocatorFree(output name) failed");
    }

    impl_->loaded = true;
    std::cout << "[ORT] Loaded model: " << model_path
              << " (" << num_inputs << " inputs, " << num_outputs << " outputs)" << std::endl;
    return true;
#else
    std::cout << "[ORT] ONNX Runtime not available, using stub" << std::endl;
    (void)device_id;
    impl_->loaded = true;
    impl_->input_names_ = {"input"};
    impl_->output_names_ = {"output"};
    return true;
#endif
}

std::vector<std::vector<float>> OrtSessionWrapper::run(
    const std::vector<std::string>& input_names,
    const std::vector<std::vector<int64_t>>& input_shapes,
    const std::vector<std::vector<float>>& input_data
) {
#ifdef HAS_ONNXRUNTIME
    const OrtApi* g = ort_api();
    if (!impl_->loaded || !impl_->session) return {};

    // H7: validate that all parallel input arrays have matching sizes.
    if (input_names.size() != input_shapes.size() ||
        input_names.size() != input_data.size()) {
        std::cerr << "[ORT] Mismatched input array sizes" << std::endl;
        return {};
    }

    std::vector<OrtValue*> inputs(input_names.size(), nullptr);
    std::vector<OrtValue*> outputs(impl_->output_names_.size(), nullptr);

    // H6: create ONE memory info for the whole loop and release it at the end.
    OrtMemoryInfo* mem_info = nullptr;
    OrtStatus* mem_status = g->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &mem_info);
    if (ort_failed(mem_status, "CreateCpuMemoryInfo failed") || !mem_info) {
        return {};
    }

    // Cleanup helper: releases already-created input tensors and the memory info.
    auto cleanup_inputs = [&]() {
        for (auto* v : inputs) {
            if (v) g->ReleaseValue(v);
        }
        if (mem_info) g->ReleaseMemoryInfo(mem_info);
    };

    // Create input tensors.
    for (size_t i = 0; i < input_names.size(); i++) {
        // H8: reject non-positive dims and use checked multiplication.
        size_t element_count = 1;
        bool bad_shape = false;
        for (auto dim : input_shapes[i]) {
            if (dim <= 0) { bad_shape = true; break; }
            if (__builtin_mul_overflow(element_count,
                                       static_cast<size_t>(dim),
                                       &element_count)) {
                bad_shape = true;
                break;
            }
        }
        if (bad_shape) {
            std::cerr << "[ORT] Invalid input shape (non-positive dim or overflow)" << std::endl;
            cleanup_inputs();
            return {};
        }

        // Verify the flat data buffer matches the declared shape.
        if (element_count != input_data[i].size()) {
            std::cerr << "[ORT] Input data size does not match shape" << std::endl;
            cleanup_inputs();
            return {};
        }

        size_t byte_count = 0;
        if (__builtin_mul_overflow(element_count, sizeof(float), &byte_count)) {
            std::cerr << "[ORT] Input byte count overflow" << std::endl;
            cleanup_inputs();
            return {};
        }

        OrtStatus* status = g->CreateTensorWithDataAsOrtValue(
            mem_info,
            const_cast<float*>(input_data[i].data()),
            byte_count,
            input_shapes[i].data(),
            input_shapes[i].size(),
            ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
            &inputs[i]
        );
        if (ort_failed(status, "CreateTensorWithDataAsOrtValue failed")) {
            cleanup_inputs();
            return {};
        }
    }

    // Convert names to const char* arrays (M15: use impl_->output_names_).
    std::vector<const char*> input_name_ptrs(input_names.size());
    std::vector<const char*> output_name_ptrs(impl_->output_names_.size());
    for (size_t i = 0; i < input_names.size(); i++) {
        input_name_ptrs[i] = input_names[i].c_str();
    }
    for (size_t i = 0; i < impl_->output_names_.size(); i++) {
        output_name_ptrs[i] = impl_->output_names_[i].c_str();
    }

    // Run inference.
    OrtStatus* status = g->Run(
        impl_->session,
        nullptr,
        input_name_ptrs.data(),
        inputs.data(),
        inputs.size(),
        output_name_ptrs.data(),
        impl_->output_names_.size(),
        outputs.data()
    );

    // Inputs are no longer needed once Run has copied/consumed them.
    cleanup_inputs();

    if (ort_failed(status, "Run failed")) {
        return {};
    }

    // Extract output data.
    std::vector<std::vector<float>> results;
    results.reserve(outputs.size());
    for (size_t i = 0; i < outputs.size(); i++) {
        if (!outputs[i]) {
            std::cerr << "[ORT] Null output tensor" << std::endl;
            // Release any remaining outputs before bailing.
            for (size_t j = i; j < outputs.size(); j++) {
                if (outputs[j]) g->ReleaseValue(outputs[j]);
            }
            return {};
        }

        // C3: query shape via type-and-shape info; check every status.
        OrtTensorTypeAndShapeInfo* info = nullptr;
        OrtStatus* s = g->GetTensorTypeAndShape(outputs[i], &info);
        if (ort_failed(s, "GetTensorTypeAndShape failed") || !info) {
            for (size_t j = i; j < outputs.size(); j++) {
                if (outputs[j]) g->ReleaseValue(outputs[j]);
            }
            return {};
        }

        size_t count = 0;
        s = g->GetTensorShapeElementCount(info, &count);
        g->ReleaseTensorTypeAndShapeInfo(info);
        if (ort_failed(s, "GetTensorShapeElementCount failed")) {
            for (size_t j = i; j < outputs.size(); j++) {
                if (outputs[j]) g->ReleaseValue(outputs[j]);
            }
            return {};
        }

        float* data = nullptr;
        s = g->GetTensorMutableData(outputs[i], reinterpret_cast<void**>(&data));
        if (ort_failed(s, "GetTensorMutableData failed") || !data) {
            for (size_t j = i; j < outputs.size(); j++) {
                if (outputs[j]) g->ReleaseValue(outputs[j]);
            }
            return {};
        }

        results.emplace_back(data, data + count);
        g->ReleaseValue(outputs[i]);
    }

    return results;
#else
    // Stub: return one empty vector per output (M15: use impl_->output_names_).
    (void)input_names;
    (void)input_shapes;
    (void)input_data;
    return std::vector<std::vector<float>>(impl_->output_names_.size());
#endif
}

std::vector<std::vector<float>> OrtSessionWrapper::run_int16(
    const std::vector<std::string>& input_names,
    const std::vector<std::vector<int64_t>>& input_shapes,
    const std::vector<std::vector<int16_t>>& input_data
) {
    // Convert int16 to float and run.
    std::vector<std::vector<float>> float_data(input_data.size());
    for (size_t i = 0; i < input_data.size(); i++) {
        float_data[i].resize(input_data[i].size());
        for (size_t j = 0; j < input_data[i].size(); j++) {
            float_data[i][j] = static_cast<float>(input_data[i][j]) / 32768.0f;
        }
    }
    return run(input_names, input_shapes, float_data);
}

bool OrtSessionWrapper::is_loaded() const {
    return impl_->loaded;
}

size_t OrtSessionWrapper::input_count() const {
    return impl_->input_names_.size();
}

size_t OrtSessionWrapper::output_count() const {
    return impl_->output_names_.size();
}

std::string OrtSessionWrapper::input_name(size_t index) const {
    if (index < impl_->input_names_.size()) return impl_->input_names_[index];
    return "";
}

std::string OrtSessionWrapper::output_name(size_t index) const {
    if (index < impl_->output_names_.size()) return impl_->output_names_[index];
    return "";
}

} // namespace szca
