#include <jetstream/domains/ml/onnx_inference/block.hh>
#include <jetstream/detail/block_impl.hh>

#include <jetstream/domains/ml/onnx_inference/module.hh>

namespace Jetstream::Blocks {

struct OnnxInferenceImpl : public Block::Impl, public DynamicConfig<Blocks::OnnxInference> {
    Result configure() override;
    Result define() override;
    Result create() override;

  protected:
    std::shared_ptr<Modules::OnnxInference> moduleConfig = std::make_shared<Modules::OnnxInference>();

  private:
    // "input" for a single-entry vector, "input_N" for multi-entry.
    static std::string portKey(const std::string& base, size_t idx, size_t total) {
        return (total == 1) ? base : (base + "_" + std::to_string(idx));
    }
};

Result OnnxInferenceImpl::configure() {
    moduleConfig->modelPath = modelPath;
    moduleConfig->inputNames = inputNames;
    moduleConfig->outputNames = outputNames;
    moduleConfig->batchSize = batchSize;
    moduleConfig->executionProvider = executionProvider;

    return Result::SUCCESS;
}

Result OnnxInferenceImpl::define() {
    for (size_t i = 0; i < inputNames.size(); ++i) {
        JST_CHECK(defineInterfaceInput(portKey("input", i, inputNames.size()),
                                       "Input",
                                       "Tensor matching the model's expected input shape."));
    }
    for (size_t i = 0; i < outputNames.size(); ++i) {
        JST_CHECK(defineInterfaceOutput(portKey("output", i, outputNames.size()),
                                        "Output",
                                        "Tensor with the model's output shape."));
    }

    JST_CHECK(defineInterfaceConfig("modelPath",
                                    "Model Path",
                                    "Filesystem path to the .onnx model file.",
                                    "filepicker:onnx"));
    JST_CHECK(defineInterfaceConfig("inputNames",
                                    "Input Names",
                                    "ONNX model input tensor name(s). One port is declared per entry.",
                                    "vector:text"));
    JST_CHECK(defineInterfaceConfig("outputNames",
                                    "Output Names",
                                    "ONNX model output tensor name(s). One port is declared per entry.",
                                    "vector:text"));
    JST_CHECK(defineInterfaceConfig("batchSize",
                                    "Batch Size",
                                    "Expected batch dimension.",
                                    "int:batches"));
    JST_CHECK(defineInterfaceConfig("executionProvider",
                                    "Execution Provider",
                                    "Execution backend for running the ONNX model.",
                                    "dropdown:cpu(CPU),coreml(Core ML),tensorrt(TensorRT)"));

    return Result::SUCCESS;
}

Result OnnxInferenceImpl::create() {
    TensorMap portMap;
    for (size_t i = 0; i < inputNames.size(); ++i) {
        const std::string key = portKey("input", i, inputNames.size());
        portMap[key] = inputs().at(key);
    }
    JST_CHECK(moduleCreate("onnx_inference", moduleConfig, portMap));

    for (size_t i = 0; i < outputNames.size(); ++i) {
        const std::string key = portKey("output", i, outputNames.size());
        JST_CHECK(moduleExposeOutput(key, {"onnx_inference", key}));
    }

    return Result::SUCCESS;
}

JST_REGISTER_BLOCK(OnnxInferenceImpl);

}  // namespace Jetstream::Blocks
