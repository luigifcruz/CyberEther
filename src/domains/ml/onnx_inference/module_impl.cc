#include "module_impl.hh"

#include <algorithm>

#include <jetstream/config.hh>

#ifdef JST_OS_WINDOWS
#include <filesystem>
#endif

namespace Jetstream::Modules {

Result OnnxInferenceImpl::define() {
    for (size_t i = 0; i < inputNames.size(); ++i) {
        JST_CHECK(defineInterfaceInput(portKey("input", i, inputNames.size())));
    }
    for (size_t i = 0; i < outputNames.size(); ++i) {
        JST_CHECK(defineInterfaceOutput(portKey("output", i, outputNames.size())));
    }
    return Result::SUCCESS;
}

Result OnnxInferenceImpl::configureSessionOptions() {
    sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
    sessionOptions.SetIntraOpNumThreads(0);  // 0 = use all available cores

    try {
        const auto providers = Ort::GetAvailableProviders();

        if (executionProvider == "coreml") {
            if (std::find(providers.begin(), providers.end(), "CoreMLExecutionProvider") == providers.end()) {
                JST_ERROR("[ONNX_INFERENCE] Core ML execution provider requested, but this ONNX Runtime build does not provide CoreMLExecutionProvider.");
                return Result::ERROR;
            }

            sessionOptions.AppendExecutionProvider("CoreMLExecutionProvider");
            JST_DEBUG("[ONNX_INFERENCE] Using Core ML execution provider.");
        } else if (executionProvider == "tensorrt") {
            if (std::find(providers.begin(), providers.end(), "TensorrtExecutionProvider") == providers.end()) {
                JST_ERROR("[ONNX_INFERENCE] TensorRT execution provider requested, but this ONNX Runtime build does not provide TensorrtExecutionProvider.");
                return Result::ERROR;
            }
            if (std::find(providers.begin(), providers.end(), "CUDAExecutionProvider") == providers.end()) {
                JST_ERROR("[ONNX_INFERENCE] TensorRT execution provider requested, but this ONNX Runtime build does not provide CUDAExecutionProvider.");
                return Result::ERROR;
            }

            OrtTensorRTProviderOptions trtOptions{};
            trtOptions.trt_max_workspace_size = 1ULL << 30;
            sessionOptions.AppendExecutionProvider_TensorRT(trtOptions);
            OrtCUDAProviderOptions cudaOptions{};
            sessionOptions.AppendExecutionProvider_CUDA(cudaOptions);
            JST_DEBUG("[ONNX_INFERENCE] Using TensorRT execution provider.");
        }
    } catch (const Ort::Exception& e) {
        JST_ERROR("[ONNX_INFERENCE] Failed to configure execution provider '{}': {}",
                  executionProvider, e.what());
        return Result::ERROR;
    }

    return Result::SUCCESS;
}

Result OnnxInferenceImpl::readModelShapes() {
    inputShapes.assign(inputTensors.size(), {});
    for (size_t i = 0; i < inputTensors.size(); ++i) {
        if (inputTensors[i].dtype() != DataType::F32) {
            JST_ERROR("[ONNX_INFERENCE] Input port '{}' must be F32, got {}.",
                      portKey("input", i, inputTensors.size()), inputTensors[i].dtype());
            return Result::ERROR;
        }
        for (const auto d : inputTensors[i].shape()) {
            inputShapes[i].push_back(static_cast<int64_t>(d));
        }
        if (inputShapes[i].empty()) {
            JST_ERROR("[ONNX_INFERENCE] Input tensor {} has an empty shape.", i);
            return Result::ERROR;
        }
    }

    outputShapes.assign(outputNames.size(), {});
    bool hasDynamicNonBatch = false;

    try {
        for (size_t i = 0; i < outputNames.size(); ++i) {
            Ort::TypeInfo inTypeInfo = session->GetInputTypeInfo(inputSessionIdx[i < inputSessionIdx.size() ? i : 0]);
            if (inTypeInfo.GetTensorTypeAndShapeInfo().GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
                JST_ERROR("[ONNX_INFERENCE] Model input '{}' must be F32.", inputNames[i < inputNames.size() ? i : 0]);
                return Result::ERROR;
            }

            Ort::TypeInfo outTypeInfo = session->GetOutputTypeInfo(outputSessionIdx[i]);
            const auto outTensorInfo = outTypeInfo.GetTensorTypeAndShapeInfo();
            if (outTensorInfo.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
                JST_ERROR("[ONNX_INFERENCE] Model output '{}' must be F32.", outputNames[i]);
                return Result::ERROR;
            }
            outputShapes[i] = outTensorInfo.GetShape();

            if (!outputShapes[i].empty() && outputShapes[i][0] < 0) {
                outputShapes[i][0] = inputShapes[0][0];
            }
            for (size_t j = 1; j < outputShapes[i].size(); ++j) {
                if (outputShapes[i][j] < 0) { hasDynamicNonBatch = true; break; }
            }
        }
    } catch (const Ort::Exception& e) {
        JST_ERROR("[ONNX_INFERENCE] Failed to read model metadata for '{}': {}", modelPath, e.what());
        return Result::ERROR;
    }

    if (hasDynamicNonBatch) {
        auto memInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        std::vector<Ort::Value> probeVals;
        for (size_t i = 0; i < inputTensors.size(); ++i) {
            std::vector<float> probeData(inputTensors[i].size(), 0.0f);
            probeVals.emplace_back(Ort::Value::CreateTensor<float>(
                memInfo, probeData.data(), probeData.size(),
                inputShapes[i].data(), inputShapes[i].size()));
        }
        try {
            auto probeResult = session->Run(Ort::RunOptions{nullptr},
                                            ortInputNames.data(), probeVals.data(), probeVals.size(),
                                            ortOutputNames.data(), ortOutputNames.size());
            for (size_t i = 0; i < outputNames.size(); ++i) {
                outputShapes[i] = probeResult[i].GetTensorTypeAndShapeInfo().GetShape();
            }
        } catch (const Ort::Exception& e) {
            JST_ERROR("[ONNX_INFERENCE] Shape probe failed for '{}': {}", modelPath, e.what());
            return Result::ERROR;
        }
    }

    for (size_t i = 0; i < outputNames.size(); ++i) {
        for (const auto d : outputShapes[i]) {
            if (d < 0) {
                JST_ERROR("[ONNX_INFERENCE] Output '{}' still has symbolic dimensions after probe.", outputNames[i]);
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
        auto* data = inputTensors[i].data<F32>();
        if (data == nullptr) {
            JST_ERROR("[ONNX_INFERENCE] Input tensor {} data is not available.", i);
            return Result::ERROR;
        }
        inputValues.emplace_back(Ort::Value::CreateTensor<float>(
            memInfo, data, inputTensors[i].size(),
            inputShapes[i].data(), inputShapes[i].size()));
    }

    outputValues.clear();
    for (size_t i = 0; i < outputTensors.size(); ++i) {
        auto* data = outputTensors[i].data<F32>();
        if (data == nullptr) {
            JST_ERROR("[ONNX_INFERENCE] Output tensor {} data is not available.", i);
            return Result::ERROR;
        }
        outputValues.emplace_back(Ort::Value::CreateTensor<float>(
            memInfo, data, outputTensors[i].size(),
            outputShapes[i].data(), outputShapes[i].size()));
    }

    return Result::SUCCESS;
}

Result OnnxInferenceImpl::create() {
    // Empty model path — wait for user to select a file.
    if (modelPath.empty()) {
        return Result::INCOMPLETE;
    }

    // Gather and validate input tensors.
    inputTensors.clear();
    for (size_t i = 0; i < inputNames.size(); ++i) {
        inputTensors.push_back(inputs().at(portKey("input", i, inputNames.size())).tensor);
    }

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
        JST_ERROR("[ONNX_INFERENCE] Failed to load ONNX model '{}': {}", modelPath, e.what());
        return Result::ERROR;
    }

    // Find each named tensor in the session by name (not by index 0).
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
                JST_ERROR("[ONNX_INFERENCE] Input '{}' not found in model '{}'.", name, modelPath);
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
                JST_ERROR("[ONNX_INFERENCE] Output '{}' not found in model '{}'.", name, modelPath);
                return Result::ERROR;
            }
        }
    } catch (const Ort::Exception& e) {
        JST_ERROR("[ONNX_INFERENCE] Failed to read model node names for '{}': {}", modelPath, e.what());
        return Result::ERROR;
    }

    JST_CHECK(readModelShapes());

    // Allocate output tensors and register them.
    outputTensors.resize(outputNames.size());
    for (size_t i = 0; i < outputNames.size(); ++i) {
        Shape shapeVec;
        shapeVec.reserve(outputShapes[i].size());
        for (const auto d : outputShapes[i]) {
            shapeVec.push_back(static_cast<U64>(d));
        }
        JST_CHECK(outputTensors[i].create(DeviceType::CPU, DataType::F32, shapeVec));

        const std::string key = portKey("output", i, outputNames.size());
        outputs()[key].produced(name(), key, outputTensors[i]);
    }

    JST_CHECK(rebuildOrtValues());

    return Result::SUCCESS;
}

Result OnnxInferenceImpl::runInference() {
    if (!session) {
        JST_ERROR("[ONNX_INFERENCE] Cannot run inference before session creation.");
        return Result::ERROR;
    }

    try {
        session->Run(Ort::RunOptions{nullptr},
                     ortInputNames.data(),  inputValues.data(),  inputValues.size(),
                     ortOutputNames.data(), outputValues.data(), outputValues.size());
    } catch (const Ort::Exception& e) {
        JST_ERROR("[ONNX_INFERENCE] Inference failed for '{}': {}", modelPath, e.what());
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
    return Result::SUCCESS;
}

Result OnnxInferenceImpl::reconfigure() {
    const auto& cfg = *candidate();

    if (cfg.modelPath         != modelPath         ||
        cfg.inputNames        != inputNames        ||
        cfg.outputNames       != outputNames       ||
        cfg.batchSize         != batchSize         ||
        cfg.executionProvider != executionProvider) {
        return Result::RECREATE;
    }

    return Result::SUCCESS;
}

}  // namespace Jetstream::Modules
