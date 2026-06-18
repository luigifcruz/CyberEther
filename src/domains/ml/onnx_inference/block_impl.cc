#include <jetstream/domains/ml/onnx_inference/block.hh>
#include <jetstream/detail/block_impl.hh>

#include <algorithm>

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
    U64 pendingModelInputCount = 1;
    U64 pendingModelOutputCount = 1;

    // "input" for a single-entry vector, "input_N" for multi-entry.
    static std::string portKey(const std::string& base, size_t idx, size_t total) {
        return (total == 1) ? base : (base + "_" + std::to_string(idx));
    }
};

Result OnnxInferenceImpl::validate() {
    const auto& config = *candidate();
    const bool inputCountChanged = config.modelInputCount != modelInputCount;
    const bool outputCountChanged = config.modelOutputCount != modelOutputCount;
    const U64 effectiveInputCount = inputCountChanged
        ? config.modelInputCount
        : std::max(config.modelInputCount, static_cast<U64>(config.inputNames.size()));
    const U64 effectiveOutputCount = outputCountChanged
        ? config.modelOutputCount
        : std::max(config.modelOutputCount, static_cast<U64>(config.outputNames.size()));

    if (effectiveInputCount == 0) {
        JST_ERROR("[BLOCK_ONNX_INFERENCE] Model input count must be greater than 0.");
        return Result::ERROR;
    }
    if (effectiveOutputCount == 0) {
        JST_ERROR("[BLOCK_ONNX_INFERENCE] Model output count must be greater than 0.");
        return Result::ERROR;
    }

    pendingModelInputCount = effectiveInputCount;
    pendingModelOutputCount = effectiveOutputCount;

    if (modelInputCount != effectiveInputCount ||
        modelOutputCount != effectiveOutputCount) {
        return Result::RECREATE;
    }

    return Result::SUCCESS;
}

Result OnnxInferenceImpl::configure() {
    modelInputCount = pendingModelInputCount;
    modelOutputCount = pendingModelOutputCount;
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
