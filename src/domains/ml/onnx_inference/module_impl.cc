#include "module_impl.hh"

#include <algorithm>

#include <jetstream/config.hh>
#include <jetstream/domains/ml/onnx_inference/onnx_types.hh>
#include <jetstream/platform.hh>

#ifdef JST_OS_WINDOWS
#include <filesystem>
#endif

namespace Jetstream::Modules {

namespace {

std::string SupportedDtypes() {
    return "F32, F64, I8, I16, I32, I64, U8, U16, U32, U64";
}

std::string DtypeMismatchMessage(const std::string& tensorName,
                                 const DataType expected,
                                 const DataType actual) {
    return jst::fmt::format(
        "[MODULE_ONNX_INFERENCE] Tensor '{}' has dtype '{}' but the model expects '{}'. "
        "Use a conversion block before ONNX Inference.",
        tensorName, actual, expected);
}

}  // namespace

Result OnnxInferenceImpl::define() {
    for (size_t i = 0; i < inputNames.size(); ++i) {
        JST_CHECK(defineInterfaceInput(jst::fmt::format("input_{}", i)));
    }
    for (size_t i = 0; i < outputNames.size(); ++i) {
        JST_CHECK(defineInterfaceOutput(jst::fmt::format("output_{}", i)));
    }
    return Result::SUCCESS;
}

Result OnnxInferenceImpl::configureSessionOptions() {
    sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
    sessionOptions.SetIntraOpNumThreads(0);

    try {
        const auto providers = Ort::GetAvailableProviders();

        if (executionProvider == "coreml") {
            if (std::find(providers.begin(), providers.end(), "CoreMLExecutionProvider") == providers.end()) {
                JST_ERROR("[MODULE_ONNX_INFERENCE] Execution provider 'coreml' requires CoreMLExecutionProvider, but it is not available in this ONNX Runtime build.");
                return Result::ERROR;
            }

            sessionOptions.AppendExecutionProvider("CoreMLExecutionProvider");
            JST_DEBUG("[MODULE_ONNX_INFERENCE] Using Core ML execution provider.");
        } else if (executionProvider == "tensorrt") {
            if (std::find(providers.begin(), providers.end(), "TensorrtExecutionProvider") == providers.end()) {
                JST_ERROR("[MODULE_ONNX_INFERENCE] Execution provider 'tensorrt' requires TensorrtExecutionProvider, but it is not available in this ONNX Runtime build.");
                return Result::ERROR;
            }
            if (std::find(providers.begin(), providers.end(), "CUDAExecutionProvider") == providers.end()) {
                JST_ERROR("[MODULE_ONNX_INFERENCE] Execution provider 'tensorrt' requires CUDAExecutionProvider, but it is not available in this ONNX Runtime build.");
                return Result::ERROR;
            }

            OrtTensorRTProviderOptions trtOptions{};
            trtOptions.trt_max_workspace_size = 1ULL << 30;
            sessionOptions.AppendExecutionProvider_TensorRT(trtOptions);
            OrtCUDAProviderOptions cudaOptions{};
            sessionOptions.AppendExecutionProvider_CUDA(cudaOptions);
            JST_DEBUG("[MODULE_ONNX_INFERENCE] Using TensorRT execution provider.");
        } else if (executionProvider != "cpu") {
            JST_ERROR("[MODULE_ONNX_INFERENCE] Unknown execution provider '{}'.", executionProvider);
            return Result::ERROR;
        }
    } catch (const Ort::Exception& e) {
        JST_ERROR("[MODULE_ONNX_INFERENCE] Failed to configure execution provider '{}': {}",
                  executionProvider, e.what());
        return Result::ERROR;
    }

    return Result::SUCCESS;
}

Result OnnxInferenceImpl::readModelShapes() {
    inputShapes.assign(inputTensors.size(), {});
    inputDtypes.assign(inputTensors.size(), DataType::None);
    for (size_t i = 0; i < inputTensors.size(); ++i) {
        const std::string inputKey = jst::fmt::format("input_{}", i);
        if (!IsSupportedOnnxInferenceTensorType(inputTensors[i].dtype())) {
            JST_ERROR("[MODULE_ONNX_INFERENCE] Input port '{}' uses unsupported dtype '{}'. Supported dtypes: {}.",
                      inputKey, inputTensors[i].dtype(), SupportedDtypes());
            return Result::ERROR;
        }
        for (const auto d : inputTensors[i].shape()) {
            inputShapes[i].push_back(static_cast<int64_t>(d));
        }
        if (inputShapes[i].empty()) {
            JST_ERROR("[MODULE_ONNX_INFERENCE] Input port '{}' has an empty shape.", inputKey);
            return Result::ERROR;
        }
    }

    outputShapes.assign(outputNames.size(), {});
    outputDtypes.assign(outputNames.size(), DataType::None);
    bool needsShapeProbe = false;

    try {
        for (size_t i = 0; i < inputNames.size(); ++i) {
            Ort::TypeInfo inTypeInfo = session->GetInputTypeInfo(inputSessionIdx[i]);
            const auto inTensorInfo = inTypeInfo.GetTensorTypeAndShapeInfo();
            const auto modelInputDtype = OnnxTensorElementTypeToDataType(inTensorInfo.GetElementType());
            if (!modelInputDtype.has_value()) {
                JST_ERROR("[MODULE_ONNX_INFERENCE] Model input '{}' uses unsupported dtype '{}' (ONNX enum {}). Supported dtypes: {}.",
                          inputNames[i], OnnxTensorElementTypeName(inTensorInfo.GetElementType()),
                          static_cast<int>(inTensorInfo.GetElementType()), SupportedDtypes());
                return Result::ERROR;
            }
            inputDtypes[i] = *modelInputDtype;
            if (inputTensors[i].dtype() != inputDtypes[i]) {
                JST_ERROR("{}", DtypeMismatchMessage(inputNames[i], inputDtypes[i], inputTensors[i].dtype()));
                return Result::ERROR;
            }
            const auto modelInputShape = inTensorInfo.GetShape();
            if (modelInputShape.size() != inputShapes[i].size()) {
                JST_ERROR("[MODULE_ONNX_INFERENCE] Invalid rank ({}) for input '{}' (expected {}).",
                          inputShapes[i].size(), inputNames[i], modelInputShape.size());
                return Result::ERROR;
            }
            for (size_t j = 0; j < modelInputShape.size(); ++j) {
                if (modelInputShape[j] >= 0 && modelInputShape[j] != inputShapes[i][j]) {
                    JST_ERROR("[MODULE_ONNX_INFERENCE] Invalid dimension {} ({}) for input '{}' (expected {}).",
                              j, inputShapes[i][j], inputNames[i], modelInputShape[j]);
                    return Result::ERROR;
                }
            }
        }

        for (size_t i = 0; i < outputNames.size(); ++i) {
            Ort::TypeInfo outTypeInfo = session->GetOutputTypeInfo(outputSessionIdx[i]);
            const auto outTensorInfo = outTypeInfo.GetTensorTypeAndShapeInfo();
            const auto modelOutputDtype = OnnxTensorElementTypeToDataType(outTensorInfo.GetElementType());
            if (!modelOutputDtype.has_value()) {
                JST_ERROR("[MODULE_ONNX_INFERENCE] Model output '{}' uses unsupported dtype '{}' (ONNX enum {}). Supported dtypes: {}.",
                          outputNames[i], OnnxTensorElementTypeName(outTensorInfo.GetElementType()),
                          static_cast<int>(outTensorInfo.GetElementType()), SupportedDtypes());
                return Result::ERROR;
            }
            outputDtypes[i] = *modelOutputDtype;
            outputShapes[i] = outTensorInfo.GetShape();

            for (size_t j = 0; j < outputShapes[i].size(); ++j) {
                if (outputShapes[i][j] >= 0) {
                    continue;
                }
                needsShapeProbe = true;
                break;
            }
        }
    } catch (const Ort::Exception& e) {
        JST_ERROR("[MODULE_ONNX_INFERENCE] Failed to read model metadata: {}", e.what());
        return Result::ERROR;
    }

    if (needsShapeProbe) {
        auto memInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        std::vector<std::vector<U8>> probeData;
        std::vector<Ort::Value> probeValues;
        probeData.reserve(inputTensors.size());
        probeValues.reserve(inputTensors.size());

        for (size_t i = 0; i < inputTensors.size(); ++i) {
            probeData.emplace_back(inputTensors[i].sizeBytes(), U8{0});
            probeValues.emplace_back(Ort::Value::CreateTensor(
                memInfo,
                probeData.back().data(),
                probeData.back().size(),
                inputShapes[i].data(),
                inputShapes[i].size(),
                DataTypeToOnnxTensorElementType(inputDtypes[i])));
        }

        try {
            auto probeResults = session->Run(Ort::RunOptions{nullptr},
                                             ortInputNames.data(), probeValues.data(), probeValues.size(),
                                             ortOutputNames.data(), ortOutputNames.size());
            for (size_t i = 0; i < outputNames.size(); ++i) {
                outputShapes[i] = probeResults[i].GetTensorTypeAndShapeInfo().GetShape();
            }
        } catch (const Ort::Exception& e) {
            JST_ERROR("[MODULE_ONNX_INFERENCE] Shape probe failed: {}", e.what());
            return Result::ERROR;
        }
    }

    for (size_t i = 0; i < outputNames.size(); ++i) {
        for (const auto dim : outputShapes[i]) {
            if (dim < 0) {
                JST_ERROR("[MODULE_ONNX_INFERENCE] Output '{}' still has symbolic dimensions after shape resolution.",
                          outputNames[i]);
                return Result::ERROR;
            }
        }
    }

    return Result::SUCCESS;
}

Result OnnxInferenceImpl::rebuildOrtValues() {
    auto memInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    inputValues.clear();
    for (size_t i = 0; i < inputTensors.size(); ++i) {
        auto* data = inputTensors[i].data();
        if (data == nullptr) {
            JST_ERROR("[MODULE_ONNX_INFERENCE] Input port '{}' data is not available.",
                      jst::fmt::format("input_{}", i));
            return Result::ERROR;
        }
        inputValues.emplace_back(Ort::Value::CreateTensor(
            memInfo,
            data,
            inputTensors[i].sizeBytes(),
            inputShapes[i].data(),
            inputShapes[i].size(),
            DataTypeToOnnxTensorElementType(inputDtypes[i])));
    }

    outputValues.clear();
    for (size_t i = 0; i < outputTensors.size(); ++i) {
        auto* data = outputTensors[i].data();
        if (data == nullptr) {
            JST_ERROR("[MODULE_ONNX_INFERENCE] Output port '{}' data is not available.",
                      jst::fmt::format("output_{}", i));
            return Result::ERROR;
        }
        outputValues.emplace_back(Ort::Value::CreateTensor(
            memInfo,
            data,
            outputTensors[i].sizeBytes(),
            outputShapes[i].data(),
            outputShapes[i].size(),
            DataTypeToOnnxTensorElementType(outputDtypes[i])));
    }

    return Result::SUCCESS;
}

Result OnnxInferenceImpl::create() {
    if (modelPath.empty()) {
        return Result::INCOMPLETE;
    }

    inputTensors.clear();
    for (size_t i = 0; i < inputNames.size(); ++i) {
        inputTensors.push_back(inputs().at(jst::fmt::format("input_{}", i)).tensor);
    }

    JST_CHECK(configureSessionOptions());

    try {
#ifdef JST_OS_WINDOWS
        const auto ortModelPath = Platform::PathFromUtf8(modelPath);
        const ORTCHAR_T* ortModelPathData = ortModelPath.c_str();
#else
        const ORTCHAR_T* ortModelPathData = modelPath.c_str();
#endif
        session = std::make_unique<Ort::Session>(ortEnv, ortModelPathData, sessionOptions);
    } catch (const Ort::Exception& e) {
        JST_ERROR("[MODULE_ONNX_INFERENCE] Failed to load ONNX model: {}", e.what());
        return Result::ERROR;
    }

    try {
        inputNameAllocs.clear();
        outputNameAllocs.clear();
        ortInputNames.clear();
        ortOutputNames.clear();
        inputSessionIdx.clear();
        outputSessionIdx.clear();

        for (const auto& name : inputNames) {
            bool found = false;
            for (size_t j = 0; j < session->GetInputCount(); ++j) {
                auto alloc = session->GetInputNameAllocated(j, allocator);
                if (std::string(alloc.get()) == name) {
                    inputSessionIdx.push_back(j);
                    ortInputNames.push_back(alloc.get());
                    inputNameAllocs.push_back(std::move(alloc));
                    found = true;
                    break;
                }
            }
            if (!found) {
                std::vector<std::string> availableInputs;
                availableInputs.reserve(session->GetInputCount());
                for (size_t j = 0; j < session->GetInputCount(); ++j) {
                    auto alloc = session->GetInputNameAllocated(j, allocator);
                    availableInputs.emplace_back(alloc.get());
                }
                JST_ERROR("[MODULE_ONNX_INFERENCE] Input '{}' not found. Available inputs: [{}].",
                          name, jst::fmt::join(availableInputs, ", "));
                return Result::ERROR;
            }
        }

        for (const auto& name : outputNames) {
            bool found = false;
            for (size_t j = 0; j < session->GetOutputCount(); ++j) {
                auto alloc = session->GetOutputNameAllocated(j, allocator);
                if (std::string(alloc.get()) == name) {
                    outputSessionIdx.push_back(j);
                    ortOutputNames.push_back(alloc.get());
                    outputNameAllocs.push_back(std::move(alloc));
                    found = true;
                    break;
                }
            }
            if (!found) {
                std::vector<std::string> availableOutputs;
                availableOutputs.reserve(session->GetOutputCount());
                for (size_t j = 0; j < session->GetOutputCount(); ++j) {
                    auto alloc = session->GetOutputNameAllocated(j, allocator);
                    availableOutputs.emplace_back(alloc.get());
                }
                JST_ERROR("[MODULE_ONNX_INFERENCE] Output '{}' not found. Available outputs: [{}].",
                          name, jst::fmt::join(availableOutputs, ", "));
                return Result::ERROR;
            }
        }
    } catch (const Ort::Exception& e) {
        JST_ERROR("[MODULE_ONNX_INFERENCE] Failed to read model input/output names: {}", e.what());
        return Result::ERROR;
    }

    JST_CHECK(readModelShapes());

    outputTensors.resize(outputNames.size());
    for (size_t i = 0; i < outputNames.size(); ++i) {
        Shape shapeVec;
        shapeVec.reserve(outputShapes[i].size());
        for (const auto d : outputShapes[i]) {
            shapeVec.push_back(static_cast<U64>(d));
        }
        JST_CHECK(outputTensors[i].create(DeviceType::CPU, outputDtypes[i], shapeVec));

        const std::string key = jst::fmt::format("output_{}", i);
        outputs()[key].produced(name(), key, outputTensors[i]);
    }

    JST_CHECK(rebuildOrtValues());

    return Result::SUCCESS;
}

Result OnnxInferenceImpl::runInference() {
    if (!session) {
        JST_ERROR("[MODULE_ONNX_INFERENCE] Cannot run inference before session creation.");
        return Result::ERROR;
    }

    try {
        const char* const* outputNames = ortOutputNames.empty() ? nullptr : ortOutputNames.data();
        Ort::Value* outputValuesPtr = outputValues.empty() ? nullptr : outputValues.data();
        session->Run(Ort::RunOptions{nullptr},
                     ortInputNames.data(), inputValues.data(), inputValues.size(),
                     outputNames, outputValuesPtr, outputValues.size());
    } catch (const Ort::Exception& e) {
        JST_ERROR("[MODULE_ONNX_INFERENCE] Inference failed: {}", e.what());
        return Result::ERROR;
    }

    return Result::SUCCESS;
}

Result OnnxInferenceImpl::destroy() {
    inputValues.clear();
    outputValues.clear();
    session.reset();
    inputNameAllocs.clear();
    outputNameAllocs.clear();
    ortInputNames.clear();
    ortOutputNames.clear();
    inputSessionIdx.clear();
    outputSessionIdx.clear();
    inputTensors.clear();
    outputTensors.clear();
    inputShapes.clear();
    outputShapes.clear();
    inputDtypes.clear();
    outputDtypes.clear();
    return Result::SUCCESS;
}

Result OnnxInferenceImpl::reconfigure() {
    const auto& cfg = *candidate();

    if (cfg.modelPath != modelPath ||
        cfg.inputNames != inputNames ||
        cfg.outputNames != outputNames ||
        cfg.executionProvider != executionProvider) {
        return Result::RECREATE;
    }

    return Result::SUCCESS;
}

}  // namespace Jetstream::Modules
