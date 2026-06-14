#include "module_impl.hh"

#include <algorithm>

#include <jetstream/config.hh>

#ifdef JST_OS_WINDOWS
#include <filesystem>
#endif

namespace Jetstream::Modules {

Result InferImpl::define() {
    JST_CHECK(defineInterfaceInput("input"));
    JST_CHECK(defineInterfaceOutput("output"));

    return Result::SUCCESS;
}

Result InferImpl::configureSessionOptions() {
    sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

    try {
        const auto providers = Ort::GetAvailableProviders();

        if (executionProvider == "coreml") {
            if (std::find(providers.begin(), providers.end(), "CoreMLExecutionProvider") == providers.end()) {
                JST_ERROR("[INFER] Core ML execution provider requested, but this ONNX Runtime build does not provide CoreMLExecutionProvider.");
                return Result::ERROR;
            }

            sessionOptions.AppendExecutionProvider("CoreMLExecutionProvider");
            JST_DEBUG("[INFER] Using Core ML execution provider.");
        } else if (executionProvider == "tensorrt") {
            if (std::find(providers.begin(), providers.end(), "TensorrtExecutionProvider") == providers.end()) {
                JST_ERROR("[INFER] TensorRT execution provider requested, but this ONNX Runtime build does not provide TensorrtExecutionProvider.");
                return Result::ERROR;
            }
            if (std::find(providers.begin(), providers.end(), "CUDAExecutionProvider") == providers.end()) {
                JST_ERROR("[INFER] TensorRT execution provider requested, but this ONNX Runtime build does not provide CUDAExecutionProvider.");
                return Result::ERROR;
            }

            OrtTensorRTProviderOptions trtOptions{};
            trtOptions.trt_max_workspace_size = 1ULL << 30;
            sessionOptions.AppendExecutionProvider_TensorRT(trtOptions);
            OrtCUDAProviderOptions cudaOptions{};
            sessionOptions.AppendExecutionProvider_CUDA(cudaOptions);
            JST_DEBUG("[INFER] Using TensorRT execution provider.");
        }
    } catch (const Ort::Exception& e) {
        JST_ERROR("[INFER] Failed to configure execution provider '{}': {}",
                  executionProvider, e.what());
        return Result::ERROR;
    }

    return Result::SUCCESS;
}

Result InferImpl::readModelShapes() {
    if (input.dtype() != DataType::F32) {
        JST_ERROR("[INFER] Input tensor must be F32, got {}.", input.dtype());
        return Result::ERROR;
    }

    // Use the upstream tensor's concrete shape rather than model metadata —
    // the model may declare symbolic dims (-1) that the tensor already resolves.
    inputShape.clear();
    for (const auto d : input.shape()) {
        inputShape.push_back(static_cast<int64_t>(d));
    }
    if (inputShape.empty()) {
        JST_ERROR("[INFER] Input tensor shape cannot be empty.");
        return Result::ERROR;
    }

    // IMPORTANT: TensorTypeAndShapeInfo holds a non-owning pointer into TypeInfo;
    // keep TypeInfo alive until GetShape() returns.
    try {
        Ort::TypeInfo inTypeInfo = session->GetInputTypeInfo(0);
        const auto inTensorInfo = inTypeInfo.GetTensorTypeAndShapeInfo();
        if (inTensorInfo.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
            JST_ERROR("[INFER] Model input must be F32.");
            return Result::ERROR;
        }

        Ort::TypeInfo outTypeInfo = session->GetOutputTypeInfo(0);
        const auto outTensorInfo = outTypeInfo.GetTensorTypeAndShapeInfo();
        if (outTensorInfo.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
            JST_ERROR("[INFER] Model output must be F32.");
            return Result::ERROR;
        }
        outputShape = outTensorInfo.GetShape();
    } catch (const Ort::Exception& e) {
        JST_ERROR("[INFER] Failed to read model metadata for '{}': {}", modelPath, e.what());
        return Result::ERROR;
    }

    if (!outputShape.empty() && outputShape[0] < 0) {
        outputShape[0] = inputShape[0];
    }

    // If non-batch dims are still symbolic, probe-run to discover the real shape.
    bool hasDynamicNonBatch = false;
    for (size_t i = 1; i < outputShape.size(); ++i) {
        if (outputShape[i] < 0) { hasDynamicNonBatch = true; break; }
    }

    if (hasDynamicNonBatch) {
        auto memInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        std::vector<float> probeData(input.size(), 0.0f);
        std::vector<Ort::Value> probeVals;
        probeVals.emplace_back(Ort::Value::CreateTensor<float>(memInfo, probeData.data(), probeData.size(),
                                                               inputShape.data(), inputShape.size()));
        try {
            auto probeResult = session->Run(Ort::RunOptions{nullptr},
                                            inputNames.data(),
                                            probeVals.data(), 1,
                                            outputNames.data(), 1);
            outputShape = probeResult[0].GetTensorTypeAndShapeInfo().GetShape();
        } catch (const Ort::Exception& e) {
            JST_ERROR("[INFER] Shape probe failed for '{}': {}", modelPath, e.what());
            return Result::ERROR;
        }
    }

    for (const auto d : outputShape) {
        if (d < 0) {
            JST_ERROR("[INFER] Output shape for '{}' still contains symbolic dimensions.", modelPath);
            return Result::ERROR;
        }
    }

    return Result::SUCCESS;
}

Result InferImpl::rebuildOrtValues() {
    auto memInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    auto* inputData = input.data<F32>();
    if (inputData == nullptr) {
        JST_ERROR("[INFER] Input tensor data is not available.");
        return Result::ERROR;
    }

    auto* outputData = output.data<F32>();
    if (outputData == nullptr) {
        JST_ERROR("[INFER] Output tensor data is not available.");
        return Result::ERROR;
    }

    inputValues.clear();
    inputValues.emplace_back(Ort::Value::CreateTensor<float>(
        memInfo, inputData, input.size(),
        inputShape.data(), inputShape.size()));

    outputValues.clear();
    outputValues.emplace_back(Ort::Value::CreateTensor<float>(
        memInfo, outputData, output.size(),
        outputShape.data(), outputShape.size()));

    return Result::SUCCESS;
}

Result InferImpl::create() {
    input = inputs().at("input").tensor;

    JST_CHECK(configureSessionOptions());

    try {
#ifdef JST_OS_WINDOWS
        const auto ortModelPath = std::filesystem::path(modelPath).wstring();
        const ORTCHAR_T* ortModelPathData = ortModelPath.c_str();
#else
        const ORTCHAR_T* ortModelPathData = modelPath.c_str();
#endif
        session = std::make_unique<Ort::Session>(ortEnv, ortModelPathData, sessionOptions);
    } catch (const Ort::Exception& e) {
        JST_ERROR("[INFER] Failed to load ONNX model '{}': {}", modelPath, e.what());
        return Result::ERROR;
    }

    try {
        inputNameAllocs.clear();
        outputNameAllocs.clear();
        inputNameAllocs.push_back(session->GetInputNameAllocated(0, allocator));
        outputNameAllocs.push_back(session->GetOutputNameAllocated(0, allocator));
        inputNames  = {inputNameAllocs[0].get()};
        outputNames = {outputNameAllocs[0].get()};
    } catch (const Ort::Exception& e) {
        JST_ERROR("[INFER] Failed to read model node names for '{}': {}", modelPath, e.what());
        return Result::ERROR;
    }

    JST_CHECK(readModelShapes());

    Shape outputShapeVec;
    outputShapeVec.reserve(outputShape.size());
    for (const auto d : outputShape) {
        outputShapeVec.push_back(static_cast<U64>(d));
    }
    JST_CHECK(output.create(DeviceType::CPU, DataType::F32, outputShapeVec));

    JST_CHECK(rebuildOrtValues());

    outputs()["output"].produced(name(), "output", output);

    return Result::SUCCESS;
}

Result InferImpl::runInference() {
    if (!session) {
        JST_ERROR("[INFER] Cannot run inference before session creation.");
        return Result::ERROR;
    }

    try {
        session->Run(Ort::RunOptions{nullptr},
                     inputNames.data(),  inputValues.data(),  inputValues.size(),
                     outputNames.data(), outputValues.data(), outputValues.size());
    } catch (const Ort::Exception& e) {
        JST_ERROR("[INFER] Inference failed for '{}': {}", modelPath, e.what());
        return Result::ERROR;
    }

    return Result::SUCCESS;
}

Result InferImpl::destroy() {
    inputValues.clear();
    outputValues.clear();
    session.reset();
    inputNameAllocs.clear();
    outputNameAllocs.clear();
    inputNames.clear();
    outputNames.clear();
    return Result::SUCCESS;
}

Result InferImpl::reconfigure() {
    const auto& cfg = *candidate();

    if (cfg.modelPath != modelPath || cfg.inputName != inputName ||
        cfg.outputName != outputName || cfg.batchSize != batchSize ||
        cfg.executionProvider != executionProvider) {
        return Result::RECREATE;
    }

    return Result::SUCCESS;
}

}  // namespace Jetstream::Modules
