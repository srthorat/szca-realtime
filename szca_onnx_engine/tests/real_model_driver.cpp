// Real-model inference driver (not a unit test — a manual verification tool).
//
// Drives an actual downloaded ONNX model through the production
// OrtSessionWrapper::run() path to prove the ORT wrapper performs real
// inference end-to-end (load -> bind inputs -> Run -> read outputs), not just
// that it compiles.
//
// Usage: real_model_driver <path-to-dfn3-enc.onnx>
// Expects the DeepFilterNet3 encoder (inputs: feat_erb[1,1,S,32],
// feat_spec[1,2,S,96]; 7 outputs incl. lsnr).

#include "ort_utils.h"

#include <cmath>
#include <iostream>
#include <vector>

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "usage: " << argv[0] << " <dfn3_enc.onnx>" << std::endl;
        return 2;
    }
    const std::string model_path = argv[1];

    // Touch the environment singleton first (creates the OrtEnv).
    if (!szca::OrtEnvironment::instance().env()) {
        std::cerr << "FAIL: ORT environment not initialized (built without ORT?)"
                  << std::endl;
        return 1;
    }

    szca::OrtSessionWrapper sess;
    if (!sess.load(model_path)) {
        std::cerr << "FAIL: could not load " << model_path << std::endl;
        return 1;
    }

    std::cout << "Loaded. inputs=" << sess.input_count()
              << " outputs=" << sess.output_count() << std::endl;
    for (size_t i = 0; i < sess.input_count(); i++)
        std::cout << "  in[" << i << "]=" << sess.input_name(i) << std::endl;
    for (size_t i = 0; i < sess.output_count(); i++)
        std::cout << "  out[" << i << "]=" << sess.output_name(i) << std::endl;

    // One time frame (S=1). feat_erb: [1,1,1,32]=32 floats; feat_spec:
    // [1,2,1,96]=192 floats. Fill with a deterministic non-zero signal.
    const int S = 1;
    std::vector<float> feat_erb(1 * 1 * S * 32);
    std::vector<float> feat_spec(1 * 2 * S * 96);
    for (size_t i = 0; i < feat_erb.size(); i++)
        feat_erb[i] = 0.01f * static_cast<float>(i % 7);
    for (size_t i = 0; i < feat_spec.size(); i++)
        feat_spec[i] = 0.005f * static_cast<float>((i % 11) - 5);

    std::vector<std::string> names = {sess.input_name(0), sess.input_name(1)};
    std::vector<std::vector<int64_t>> shapes = {
        {1, 1, S, 32},
        {1, 2, S, 96},
    };
    std::vector<std::vector<float>> data = {feat_erb, feat_spec};

    auto outputs = sess.run(names, shapes, data);
    if (outputs.empty()) {
        std::cerr << "FAIL: run() returned no outputs" << std::endl;
        return 1;
    }

    std::cout << "run() produced " << outputs.size() << " output tensors:"
              << std::endl;
    bool any_finite_nonzero = false;
    for (size_t i = 0; i < outputs.size(); i++) {
        const auto& o = outputs[i];
        double sum = 0.0;
        bool all_finite = true;
        for (float v : o) {
            if (!std::isfinite(v)) all_finite = false;
            sum += v;
            if (std::isfinite(v) && v != 0.0f) any_finite_nonzero = true;
        }
        std::cout << "  out[" << i << "] " << sess.output_name(i)
                  << " count=" << o.size()
                  << " sum=" << sum
                  << " finite=" << (all_finite ? "yes" : "NO")
                  << (o.empty() ? "" : "")
                  << std::endl;
        if (!all_finite) {
            std::cerr << "FAIL: non-finite values in output " << i << std::endl;
            return 1;
        }
    }

    if (!any_finite_nonzero) {
        std::cerr << "FAIL: all outputs were zero — inference likely did not run"
                  << std::endl;
        return 1;
    }

    std::cout << "PASS: real DFN3 encoder inference produced finite, "
                 "non-trivial outputs." << std::endl;
    return 0;
}
