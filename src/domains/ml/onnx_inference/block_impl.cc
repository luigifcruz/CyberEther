#include <jetstream/domains/ml/onnx_inference/block.hh>
#include <jetstream/detail/block_impl.hh>

#include <jetstream/config.hh>
#include <jetstream/domains/ml/onnx_inference/module.hh>
#include <jetstream/domains/ml/onnx_inference/onnx_types.hh>

#ifdef JST_OS_WINDOWS
#include <filesystem>
#endif

#include <algorithm>

#include <onnxruntime_cxx_api.h>

namespace Jetstream::Blocks {

namespace {

bool HasProvider(const std::vector<std::string>& providers, const std::string& name) {
    return std::find(providers.begin(), providers.end(), name) != providers.end();
}

bool IsExecutionProvider(const std::string& name) {
    return name == "cpu" || name == "coreml" || name == "tensorrt";
}

std::string FormatShape(const std::vector<int64_t>& shape) {
    std::vector<std::string> dims;
    dims.reserve(shape.size());
    for (const auto dim : shape) {
        dims.push_back(dim < 0 ? "?" : jst::fmt::format("{}", dim));
    }
    return jst::fmt::format("[{}]", jst::fmt::join(dims, ", "));
}

std::string ExecutionProviderDropdown() {
    std::vector<std::string> options{"cpu(CPU)"};

    try {
        const auto providers = Ort::GetAvailableProviders();
        if (HasProvider(providers, "CoreMLExecutionProvider")) {
            options.emplace_back("coreml(Core ML)");
        }
        if (HasProvider(providers, "TensorrtExecutionProvider") &&
            HasProvider(providers, "CUDAExecutionProvider")) {
            options.emplace_back("tensorrt(TensorRT)");
        }
    } catch (const Ort::Exception& e) {
        JST_ERROR("[BLOCK_ONNX_INFERENCE] Failed to enumerate ONNX Runtime providers: {}", e.what());
    }

    return jst::fmt::format("dropdown:{}", jst::fmt::join(options, ","));
}

}  // namespace

struct OnnxInferenceImpl : public Block::Impl, public DynamicConfig<Blocks::OnnxInference> {
    Result validate() override;
    Result configure() override;
    Result define() override;
    Result create() override;

  protected:
    std::shared_ptr<Modules::OnnxInference> moduleConfig = std::make_shared<Modules::OnnxInference>();
    std::vector<std::string> inputNames;
    std::vector<std::vector<int64_t>> inputShapes;
    std::vector<DataType> inputDtypes;
    std::vector<std::string> outputNames;
    std::vector<DataType> outputDtypes;
    bool unsupportedModelDtypes = false;

  private:
    Result readModelTensorNames();
};

Result OnnxInferenceImpl::readModelTensorNames() {
    inputNames.clear();
    inputShapes.clear();
    inputDtypes.clear();
    outputNames.clear();
    outputDtypes.clear();
    unsupportedModelDtypes = false;

    if (modelPath.empty()) {
        return Result::SUCCESS;
    }

    try {
        Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "jetstream-onnx-metadata"};
        Ort::SessionOptions sessionOptions;

#ifdef JST_OS_WINDOWS
        const auto ortModelPath = std::filesystem::path(modelPath).wstring();
        const ORTCHAR_T* ortModelPathData = ortModelPath.c_str();
#else
        const ORTCHAR_T* ortModelPathData = modelPath.c_str();
#endif
        Ort::Session session(env, ortModelPathData, sessionOptions);
        Ort::AllocatorWithDefaultOptions allocator;

        const auto inputCount = session.GetInputCount();
        if (inputCount == 0) {
            JST_ERROR("[BLOCK_ONNX_INFERENCE] Model does not expose any inputs.");
            return Result::ERROR;
        }
        inputNames.reserve(inputCount);
        inputShapes.reserve(inputCount);
        inputDtypes.reserve(inputCount);
        for (size_t i = 0; i < inputCount; ++i) {
            auto name = session.GetInputNameAllocated(i, allocator);
            inputNames.emplace_back(name.get());
            // NOTE: GetTensorTypeAndShapeInfo() returns a non-owning view into the
            // Ort::TypeInfo, so the TypeInfo must outlive every read below.
            Ort::TypeInfo typeInfo = session.GetInputTypeInfo(i);
            const auto tensorInfo = typeInfo.GetTensorTypeAndShapeInfo();
            const auto elementType = tensorInfo.GetElementType();
            const auto dtype = OnnxTensorElementTypeToDataType(elementType);
            if (!dtype.has_value()) {
                JST_WARN("[BLOCK_ONNX_INFERENCE] Model input '{}' uses unsupported dtype '{}' (ONNX enum {}). The block will still expose its ports, but the runtime may reject the model until the dtype is supported.",
                         inputNames.back(),
                         OnnxTensorElementTypeName(elementType),
                         static_cast<int>(elementType));
                inputShapes.push_back(tensorInfo.GetShape());
                inputDtypes.push_back(DataType::None);
                unsupportedModelDtypes = true;
                continue;
            }
            inputShapes.push_back(tensorInfo.GetShape());
            inputDtypes.push_back(*dtype);
        }

        const auto outputCount = session.GetOutputCount();
        if (outputCount == 0) {
            JST_WARN("[BLOCK_ONNX_INFERENCE] Model does not expose any outputs. Input ports will still be exposed.");
            return Result::SUCCESS;
        }
        outputNames.reserve(outputCount);
        outputDtypes.reserve(outputCount);
        for (size_t i = 0; i < outputCount; ++i) {
            auto name = session.GetOutputNameAllocated(i, allocator);
            outputNames.emplace_back(name.get());
            // NOTE: GetTensorTypeAndShapeInfo() returns a non-owning view into the
            // Ort::TypeInfo, so the TypeInfo must outlive every read below.
            Ort::TypeInfo typeInfo = session.GetOutputTypeInfo(i);
            const auto tensorInfo = typeInfo.GetTensorTypeAndShapeInfo();
            const auto elementType = tensorInfo.GetElementType();
            const auto dtype = OnnxTensorElementTypeToDataType(elementType);
            if (!dtype.has_value()) {
                JST_WARN("[BLOCK_ONNX_INFERENCE] Model output '{}' uses unsupported dtype '{}' (ONNX enum {}). The block will still expose its ports, but the runtime may reject the model until the dtype is supported.",
                         outputNames.back(),
                         OnnxTensorElementTypeName(elementType),
                         static_cast<int>(elementType));
                outputDtypes.push_back(DataType::None);
                unsupportedModelDtypes = true;
                continue;
            }
            outputDtypes.push_back(*dtype);
        }
    } catch (const Ort::Exception& e) {
        JST_ERROR("[BLOCK_ONNX_INFERENCE] Failed to read ONNX model metadata: {}", e.what());
        return Result::ERROR;
    }

    return Result::SUCCESS;
}

Result OnnxInferenceImpl::validate() {
    const auto& config = *candidate();

    if (!IsExecutionProvider(config.executionProvider)) {
        JST_ERROR("[BLOCK_ONNX_INFERENCE] Unknown execution provider '{}'.", config.executionProvider);
        return Result::ERROR;
    }

    if (modelPath != config.modelPath ||
        executionProvider != config.executionProvider) {
        return Result::RECREATE;
    }

    return Result::SUCCESS;
}

Result OnnxInferenceImpl::configure() {
    if (readModelTensorNames() != Result::SUCCESS) {
        inputNames.clear();
        inputShapes.clear();
        inputDtypes.clear();
        outputNames.clear();
        outputDtypes.clear();
    }

    moduleConfig->modelPath = modelPath;
    moduleConfig->inputNames = inputNames;
    moduleConfig->outputNames = outputNames;
    moduleConfig->executionProvider = executionProvider;

    return Result::SUCCESS;
}

Result OnnxInferenceImpl::define() {
    for (size_t i = 0; i < inputNames.size(); ++i) {
        JST_CHECK(defineInterfaceInput(jst::fmt::format("input_{}", i),
                                       inputNames[i],
                                       jst::fmt::format("ONNX model input tensor '{}'. Expected shape: {}, dtype: {}.",
                                                        inputNames[i], FormatShape(inputShapes[i]), inputDtypes[i])));
    }
    for (size_t i = 0; i < outputNames.size(); ++i) {
        JST_CHECK(defineInterfaceOutput(jst::fmt::format("output_{}", i),
                                        outputNames[i],
                                        jst::fmt::format("ONNX model output tensor '{}'. dtype: {}.",
                                                        outputNames[i], outputDtypes[i])));
    }

    JST_CHECK(defineInterfaceConfig("modelPath",
                                    "Model Path",
                                    "Filesystem path to the .onnx model file.",
                                    "filepicker:onnx"));
    JST_CHECK(defineInterfaceConfig("executionProvider",
                                    "Execution Provider",
                                    "Execution backend for running the ONNX model.",
                                    ExecutionProviderDropdown()));

    return Result::SUCCESS;
}

Result OnnxInferenceImpl::create() {
    if (modelPath.empty() || inputNames.empty()) {
        return Result::INCOMPLETE;
    }

    if (unsupportedModelDtypes) {
        JST_WARN("[BLOCK_ONNX_INFERENCE] Skipping runtime module creation because the model metadata contains unsupported dtypes. The block ports remain visible for wiring, but inference is disabled until a supported model is selected.");
        return Result::INCOMPLETE;
    }

    TensorMap portMap;
    for (size_t i = 0; i < inputNames.size(); ++i) {
        const std::string key = jst::fmt::format("input_{}", i);
        portMap[key] = inputs().at(key);
    }
    JST_CHECK(moduleCreate("onnx_inference", moduleConfig, portMap));

    for (size_t i = 0; i < outputNames.size(); ++i) {
        const std::string key = jst::fmt::format("output_{}", i);
        JST_CHECK(moduleExposeOutput(key, {"onnx_inference", key}));
    }

    return Result::SUCCESS;
}

JST_REGISTER_BLOCK(OnnxInferenceImpl);

}  // namespace Jetstream::Blocks
