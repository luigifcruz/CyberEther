#include "module_impl.hh"

#include <jetstream/config.hh>

#ifdef JETSTREAM_LOADER_ONNXRUNTIME_COREML_AVAILABLE
#include <coreml_provider_factory.h>
#endif

namespace Jetstream::Modules {

Result InferImpl::define() {
    JST_CHECK(defineInterfaceInput("input"));
    JST_CHECK(defineInterfaceOutput("output"));
    return Result::SUCCESS;
}

Result InferImpl::configureSessionOptions() {
    sessionOptions.SetIntraOpNumThreads(1);
    sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

    if (executionProvider == "coreml") {
#ifdef JETSTREAM_LOADER_ONNXRUNTIME_COREML_AVAILABLE
        const uint32_t coremlFlags = 0;
        Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_CoreML(
            sessionOptions, coremlFlags));
        JST_DEBUG("[INFER] Using CoreML execution provider.");
#else
        JST_WARN("[INFER] CoreML requested but not available on this platform — falling back to CPU.");
#endif
    } else if (executionProvider == "cuda_trt") {
#ifdef JETSTREAM_LOADER_CUDA_AVAILABLE
        OrtTensorRTProviderOptions trtOptions{};
        trtOptions.trt_max_workspace_size = 1ULL << 30;
        sessionOptions.AppendExecutionProvider_TensorRT(trtOptions);
        OrtCUDAProviderOptions cudaOptions{};
        sessionOptions.AppendExecutionProvider_CUDA(cudaOptions);
        JST_DEBUG("[INFER] Using TensorRT + CUDA execution provider.");
#else
        JST_WARN("[INFER] CUDA/TRT requested but not compiled in — falling back to CPU.");
#endif
    }

    return Result::SUCCESS;
}

Result InferImpl::readModelShapes() {
    // Use the upstream tensor's concrete shape rather than model metadata —
    // the model may declare symbolic dims (-1) that the tensor already resolves.
    inputShape.clear();
    for (const auto d : input.shape()) {
        inputShape.push_back(static_cast<int64_t>(d));
    }

    // Read declared output shape.
    // IMPORTANT: TensorTypeAndShapeInfo holds a non-owning pointer into TypeInfo;
    // keep TypeInfo alive until GetShape() returns.
    {
        Ort::TypeInfo outTypeInfo = session->GetOutputTypeInfo(0);
        outputShape = outTypeInfo.GetTensorTypeAndShapeInfo().GetShape();
    }

    // Replace batch dim (dim 0) with configured batchSize if still symbolic.
    if (!outputShape.empty() && outputShape[0] < 0) {
        outputShape[0] = static_cast<int64_t>(batchSize);
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
        probeVals.emplace_back(Ort::Value::CreateTensor<float>(
            memInfo, probeData.data(), probeData.size(),
            inputShape.data(), inputShape.size()));
        try {
            auto probeResult = session->Run(Ort::RunOptions{nullptr},
                                            inputNames.data(), probeVals.data(), 1,
                                            outputNames.data(), 1);
            outputShape = probeResult[0].GetTensorTypeAndShapeInfo().GetShape();
        } catch (const Ort::Exception& e) {
            JST_ERROR("[INFER] Shape probe failed for '{}': {}", modelPath, e.what());
            return Result::ERROR;
        }
    }

    return Result::SUCCESS;
}

Result InferImpl::rebuildOrtValues() {
    auto memInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    inputValues.clear();
    inputValues.emplace_back(Ort::Value::CreateTensor<float>(
        memInfo, input.data<F32>(), input.size(),
        inputShape.data(), inputShape.size()));

    outputValues.clear();
    outputValues.emplace_back(Ort::Value::CreateTensor<float>(
        memInfo, output.data<F32>(), output.size(),
        outputShape.data(), outputShape.size()));

    return Result::SUCCESS;
}

Result InferImpl::create() {
    input = inputs().at("input").tensor;

    JST_CHECK(configureSessionOptions());

    try {
        session = std::make_unique<Ort::Session>(ortEnv,
                                                 modelPath.c_str(),
                                                 sessionOptions);
    } catch (const Ort::Exception& e) {
        JST_ERROR("[INFER] Failed to load ONNX model '{}': {}", modelPath, e.what());
        return Result::ERROR;
    }

    // Cache node names before readModelShapes — the probe inference needs them.
    inputNameAllocs.clear();
    outputNameAllocs.clear();
    inputNameAllocs.push_back(session->GetInputNameAllocated(0, allocator));
    outputNameAllocs.push_back(session->GetOutputNameAllocated(0, allocator));
    inputNames  = {inputNameAllocs[0].get()};
    outputNames = {outputNameAllocs[0].get()};

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

    if (cfg.modelPath        != modelPath        ||
        cfg.inputName        != inputName         ||
        cfg.outputName       != outputName        ||
        cfg.batchSize        != batchSize         ||
        cfg.executionProvider != executionProvider) {
        return Result::RECREATE;
    }

    return Result::SUCCESS;
}

}  // namespace Jetstream::Modules
