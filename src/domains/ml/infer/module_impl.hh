#ifndef JETSTREAM_DOMAINS_ML_INFER_MODULE_IMPL_HH
#define JETSTREAM_DOMAINS_ML_INFER_MODULE_IMPL_HH

#include <memory>
#include <string>
#include <vector>

#include <jetstream/domains/ml/infer/module.hh>
#include <jetstream/detail/module_impl.hh>

#include <onnxruntime_cxx_api.h>

namespace Jetstream::Modules {

struct InferImpl : public Module::Impl, public DynamicConfig<Infer> {
 public:
    Result define() override;
    Result create() override;
    Result destroy() override;
    Result reconfigure() override;

 protected:
    Tensor input;
    Tensor output;

    Ort::Env ortEnv{ORT_LOGGING_LEVEL_WARNING, "jetstream"};
    Ort::SessionOptions sessionOptions;
    Ort::AllocatorWithDefaultOptions allocator;
    std::unique_ptr<Ort::Session> session;

    std::vector<Ort::AllocatedStringPtr> inputNameAllocs;
    std::vector<Ort::AllocatedStringPtr> outputNameAllocs;
    std::vector<const char*> inputNames;
    std::vector<const char*> outputNames;

    std::vector<int64_t> inputShape;
    std::vector<int64_t> outputShape;
    std::vector<Ort::Value> inputValues;
    std::vector<Ort::Value> outputValues;

    Result runInference();

  private:
    Result configureSessionOptions();
    Result readModelShapes();
    Result rebuildOrtValues();
};

}  // namespace Jetstream::Modules

#endif  // JETSTREAM_DOMAINS_ML_INFER_MODULE_IMPL_HH
