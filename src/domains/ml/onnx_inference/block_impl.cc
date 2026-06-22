#include <jetstream/domains/ml/onnx_inference/block.hh>
#include <jetstream/detail/block_impl.hh>

#include <jetstream/domains/ml/onnx_inference/module.hh>

namespace Jetstream::Blocks {

struct OnnxInferenceImpl : public Block::Impl, public DynamicConfig<Blocks::OnnxInference> {
    Result validate() override;
    Result configure() override;
    Result define() override;
    Result create() override;

  protected:
    std::shared_ptr<Modules::OnnxInference> moduleConfig = std::make_shared<Modules::OnnxInference>();

  private:
    static std::string portKey(const std::string& base, size_t idx) {
        return base + "_" + std::to_string(idx);
    }
};

Result OnnxInferenceImpl::validate() {
    const auto& config = *candidate();

    if (config.modelInputCount == 0) {
        JST_ERROR("[BLOCK_ONNX_INFERENCE] Model input count must be greater than 0.");
        return Result::ERROR;
    }
    if (config.modelOutputCount == 0) {
        JST_ERROR("[BLOCK_ONNX_INFERENCE] Model output count must be greater than 0.");
        return Result::ERROR;
    }

    if (modelInputCount != config.modelInputCount ||
        modelOutputCount != config.modelOutputCount) {
        return Result::RECREATE;
    }

    return Result::SUCCESS;
}

Result OnnxInferenceImpl::configure() {
    inputNames.resize(modelInputCount);
    outputNames.resize(modelOutputCount);

    moduleConfig->modelPath = modelPath;
    moduleConfig->inputNames = inputNames;
    moduleConfig->outputNames = outputNames;
    moduleConfig->batchSize = batchSize;
    moduleConfig->executionProvider = executionProvider;

    return Result::SUCCESS;
}

Result OnnxInferenceImpl::define() {
    for (size_t i = 0; i < inputNames.size(); ++i) {
        JST_CHECK(defineInterfaceInput(portKey("input", i),
                                       "Input",
                                       "Tensor matching the model's expected input shape."));
    }
    for (size_t i = 0; i < outputNames.size(); ++i) {
        JST_CHECK(defineInterfaceOutput(portKey("output", i),
                                        "Output",
                                        "Tensor with the model's output shape."));
    }

    JST_CHECK(defineInterfaceConfig("modelPath",
                                    "Model Path",
                                    "Filesystem path to the .onnx model file.",
                                    "filepicker:onnx"));
    JST_CHECK(defineInterfaceConfig("modelInputCount",
                                    "Model Input Count",
                                    "Number of ONNX model input tensors.",
                                    "int:tensors"));
    JST_CHECK(defineInterfaceConfig("modelOutputCount",
                                    "Model Output Count",
                                    "Number of ONNX model output tensors.",
                                    "int:tensors"));
    JST_CHECK(defineInterfaceConfig("inputNames",
                                    "Model Input Tensor",
                                    "ONNX model input tensor name(s). One port is declared per entry.",
                                    "vector:text"));
    JST_CHECK(defineInterfaceConfig("outputNames",
                                    "Model Output Tensor",
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
        const std::string key = portKey("input", i);
        portMap[key] = inputs().at(key);
    }
    JST_CHECK(moduleCreate("onnx_inference", moduleConfig, portMap));

    for (size_t i = 0; i < outputNames.size(); ++i) {
        const std::string key = portKey("output", i);
        JST_CHECK(moduleExposeOutput(key, {"onnx_inference", key}));
    }

    return Result::SUCCESS;
}

JST_REGISTER_BLOCK(OnnxInferenceImpl);

}  // namespace Jetstream::Blocks
