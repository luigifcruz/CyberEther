#ifndef JETSTREAM_DOMAINS_ML_ONNX_INFERENCE_MODULE_IMPL_HH
#define JETSTREAM_DOMAINS_ML_ONNX_INFERENCE_MODULE_IMPL_HH

#include <memory>
#include <string>
#include <vector>

#include <jetstream/domains/ml/onnx_inference/module.hh>
#include <jetstream/detail/module_impl.hh>

#include <onnxruntime_cxx_api.h>

namespace Jetstream::Modules {

struct OnnxInferenceImpl : public Module::Impl, public DynamicConfig<OnnxInference> {
 public:
    Result define() override;
    Result create() override;
    Result destroy() override;
    Result reconfigure() override;

    Result runInference();

 protected:
    // One tensor per port (parallel to inputNames / outputNames config vectors).
    std::vector<Tensor> inputTensors;
    std::vector<Tensor> outputTensors;

    Ort::Env ortEnv{ORT_LOGGING_LEVEL_WARNING, "jetstream"};
    Ort::SessionOptions sessionOptions;
    Ort::AllocatorWithDefaultOptions allocator;
    std::unique_ptr<Ort::Session> session;

    // Raw ORT name pointers — kept alive by the alloc objects below.
    std::vector<const char*> ortInputNames;
    std::vector<const char*> ortOutputNames;
    std::vector<Ort::AllocatedStringPtr> inputNameAllocs;
    std::vector<Ort::AllocatedStringPtr> outputNameAllocs;
    // Session-internal index for each configured name (used for GetTypeInfo).
    std::vector<size_t> inputSessionIdx;
    std::vector<size_t> outputSessionIdx;

    std::vector<std::vector<int64_t>> inputShapes;
    std::vector<std::vector<int64_t>> outputShapes;
    std::vector<Ort::Value> inputValues;
    std::vector<Ort::Value> outputValues;

  private:
    Result configureSessionOptions();
    Result readModelShapes();
    Result rebuildOrtValues();

    static std::string portKey(const std::string& base, size_t idx) {
        return base + "_" + std::to_string(idx);
    }
};

}  // namespace Jetstream::Modules

#endif  // JETSTREAM_DOMAINS_ML_ONNX_INFERENCE_MODULE_IMPL_HH
