#ifndef JETSTREAM_DOMAINS_ML_ONNX_INFERENCE_MODULE_HH
#define JETSTREAM_DOMAINS_ML_ONNX_INFERENCE_MODULE_HH

#include <jetstream/module.hh>
#include <jetstream/types.hh>

namespace Jetstream::Modules {

struct OnnxInference : public Module::Config {
    std::string modelPath = "";
    std::vector<std::string> inputNames = {"modelInput"};
    std::vector<std::string> outputNames = {"output"};
    U64 batchSize = 1;
    std::string executionProvider = "cpu";

    JST_MODULE_TYPE(onnx_inference);
    JST_MODULE_PARAMS(modelPath, inputNames, outputNames, batchSize, executionProvider);
};

}  // namespace Jetstream::Modules

#endif  // JETSTREAM_DOMAINS_ML_ONNX_INFERENCE_MODULE_HH
