#include <algorithm>
#include <cstdint>
#include <limits>
#include <vector>

#include <jetstream/backend/devices/cuda/helpers.hh>
#include <jetstream/module_context.hh>
#include <jetstream/registry.hh>
#include <jetstream/runtime_context_native_cuda.hh>
#include <jetstream/scheduler_context.hh>
#include <jetstream/tools/numeric.hh>

#include "module_impl.hh"

namespace Jetstream::Modules {

namespace {

constexpr U64 kThreadsPerBlock = 256;
constexpr U64 kMaxGridSizeX = std::numeric_limits<I32>::max();
constexpr const char* kLineplotKernelName = "lineplot_update";
constexpr const char* kLineplotWaterfallKernelName = "lineplot_waterfall_update";
constexpr const char* kLineplotKernelSource = R"(
extern "C" __global__ void lineplot_update(const float* input,
                                           float* signalPoints,
                                           float* averagingBuffer,
                                           float* maxHoldPoints,
                                           unsigned long long numberOfElements,
                                           unsigned long long numberOfBatches,
                                           unsigned long long inputBatchStride,
                                           unsigned long long inputSampleStride,
                                           unsigned long long decimation,
                                           float normalizationFactor,
                                           unsigned long long averaging,
                                           unsigned int maxHoldEnabled) {
    const unsigned long long index =
        (static_cast<unsigned long long>(blockIdx.x) * blockDim.x) + threadIdx.x;
    if (index >= numberOfElements) {
        return;
    }

    float sum = 0.0f;
    for (unsigned long long batch = 0; batch < numberOfBatches; ++batch) {
        sum += input[(batch * inputBatchStride) +
                     (index * decimation * inputSampleStride)];
    }

    const float amplitude = fminf(
        fmaxf((sum * normalizationFactor) - 1.0f, -1.0f),
        1.0f);
    float average = averagingBuffer[index];
    average -= average / static_cast<float>(averaging);
    average += amplitude / static_cast<float>(averaging);

    averagingBuffer[index] = average;
    signalPoints[(index * 2) + 1] = average;

    if (maxHoldEnabled != 0) {
        float& maxValue = maxHoldPoints[(index * 2) + 1];
        if (average > maxValue) {
            maxValue = average;
        }
    }
}
)";

constexpr const char* kLineplotWaterfallKernelSource = R"(
extern "C" __global__ void
lineplot_waterfall_update(const float* input,
                          float* waterfallBins,
                          unsigned long long sampleSize,
                          unsigned long long inputSampleStride,
                          unsigned long long inputBatchStride,
                          unsigned long long retainedBatches,
                          unsigned long long height,
                          unsigned long long sourceRow,
                          unsigned long long destinationRow) {
    const unsigned long long index =
        (static_cast<unsigned long long>(blockIdx.x) * blockDim.x) + threadIdx.x;
    const unsigned long long elementCount = retainedBatches * sampleSize;
    if (index >= elementCount) {
        return;
    }

    const unsigned long long retainedBatch = index / sampleSize;
    const unsigned long long sample = index % sampleSize;
    const unsigned long long sourceBatch = sourceRow + retainedBatch;
    const unsigned long long destinationBatch =
        (destinationRow + retainedBatch) % height;

    waterfallBins[(destinationBatch * sampleSize) + sample] =
        input[(sourceBatch * inputBatchStride) +
              (sample * inputSampleStride)];
}
)";

}  // namespace

struct SignalViewImplNativeCuda : public SignalViewImpl,
                                  public NativeCudaRuntimeContext,
                                  public Scheduler::Context {
 public:
    Result validate() final;
    Result create() final;

    Result presentInitialize() override;
    Result presentSubmit() override;
    Result computeInitialize() override;
    Result computeSubmit(const cudaStream_t& stream) override;
    Result computeDeinitialize() override;

    Buffer::Config renderStateBufferConfig() const override;
    Result resetAveragingState() override;

 private:
    Tensor averagingBuffer;
    U64 lineplotGridSize = 0;
    U64 waterfallGridSize = 0;
    U64 validatedLineplotGridSize = 0;
    U64 validatedWaterfallGridSize = 0;
    bool lineplotKernelCreated = false;
    bool waterfallKernelCreated = false;
};

Result SignalViewImplNativeCuda::validate() {
    validatedLineplotGridSize = 0;
    validatedWaterfallGridSize = 0;
    JST_CHECK(SignalViewImpl::validate());

    if (!inputs().contains("signal")) {
        return Result::SUCCESS;
    }

    const Tensor& inputTensor = inputs().at("signal").tensor;
    if (!inputTensor.validShape() || inputTensor.size() == 0) {
        return Result::SUCCESS;
    }

    if (inputTensor.dtype() != DataType::F32) {
        JST_ERROR("[MODULE_SIGNAL_VIEW_NATIVE_CUDA] Unsupported input data type: {}.",
                  inputTensor.dtype());
        return Result::ERROR;
    }

    if (validatedLineplotEnabled) {
        validatedLineplotGridSize =
            validatedNumberOfElements / kThreadsPerBlock +
            (validatedNumberOfElements % kThreadsPerBlock != 0);
        if (validatedLineplotGridSize > kMaxGridSizeX) {
            JST_ERROR("[MODULE_SIGNAL_VIEW_NATIVE_CUDA] Lineplot size exceeds "
                      "the CUDA grid limit.");
            return Result::ERROR;
        }
    }

    if (validatedWaterfallEnabled) {
        U64 workItems = 0;
        const U64 retainedBatches = std::min(validatedNumberOfBatches,
                                             candidate()->waterfallHeight);
        if (!Jetstream::detail::CheckedMultiply(retainedBatches,
                                                validatedInputSampleSize,
                                                workItems)) {
            JST_ERROR("[MODULE_SIGNAL_VIEW_NATIVE_CUDA] Waterfall work size "
                      "exceeds the supported range.");
            return Result::ERROR;
        }
        validatedWaterfallGridSize =
            workItems / kThreadsPerBlock +
            (workItems % kThreadsPerBlock != 0);
        if (validatedWaterfallGridSize > kMaxGridSizeX) {
            JST_ERROR("[MODULE_SIGNAL_VIEW_NATIVE_CUDA] Waterfall size exceeds "
                      "the CUDA grid limit.");
            return Result::ERROR;
        }
    }

    return Result::SUCCESS;
}

Result SignalViewImplNativeCuda::create() {
    JST_CHECK(SignalViewImpl::create());

    lineplotGridSize = validatedLineplotGridSize;
    waterfallGridSize = validatedWaterfallGridSize;

    if (lineplotEnabled) {
        JST_CHECK(averagingBuffer.create(device(), DataType::F32, {numberOfElements}));

        std::vector<F32> initialPoints(signalPoints.size(), 0.0f);
        std::vector<F32> initialMaxHoldPoints(maxHoldPoints.size(), 0.0f);
        detail::InitializeLineplotPoints(initialPoints.data(),
                                         initialMaxHoldPoints.data(),
                                         numberOfElements);

        auto* signalBase = static_cast<std::uint8_t*>(signalPoints.buffer().data());
        auto* maxHoldBase = static_cast<std::uint8_t*>(maxHoldPoints.buffer().data());
        if (!signalBase || !maxHoldBase) {
            JST_ERROR("[MODULE_SIGNAL_VIEW_NATIVE_CUDA] Missing lineplot "
                      "state buffer.");
            return Result::ERROR;
        }

        JST_CUDA_CHECK(cudaMemcpy(signalBase + signalPoints.offsetBytes(),
                                  initialPoints.data(),
                                  signalPoints.sizeBytes(),
                                  cudaMemcpyHostToDevice), [&] {
            JST_ERROR("[MODULE_SIGNAL_VIEW_NATIVE_CUDA] Failed to "
                      "initialize signal points: {}.",
                      err);
        });

        JST_CUDA_CHECK(cudaMemcpy(maxHoldBase + maxHoldPoints.offsetBytes(),
                                  initialMaxHoldPoints.data(),
                                  maxHoldPoints.sizeBytes(),
                                  cudaMemcpyHostToDevice), [&] {
            JST_ERROR("[MODULE_SIGNAL_VIEW_NATIVE_CUDA] Failed to "
                      "initialize max hold points: {}.",
                      err);
        });
    }

    return Result::SUCCESS;
}

Result SignalViewImplNativeCuda::presentInitialize() {
    return createPresent();
}

Buffer::Config SignalViewImplNativeCuda::renderStateBufferConfig() const {
    // TODO: Restore CUDA/Vulkan zero-copy after adding cross-API synchronization.
    return {.hostAccessible = true};
}

Result SignalViewImplNativeCuda::resetAveragingState() {
    if (!lineplotEnabled) {
        return Result::SUCCESS;
    }
    void* data = averagingBuffer.buffer().data();
    if (!data) {
        JST_ERROR("[MODULE_SIGNAL_VIEW_NATIVE_CUDA] Missing averaging state buffer.");
        return Result::ERROR;
    }
    JST_CUDA_CHECK(cudaMemset(data, 0, averagingBuffer.sizeBytes()), [&] {
        JST_ERROR("[MODULE_SIGNAL_VIEW_NATIVE_CUDA] Failed to clear "
                  "averaging state: {}.",
                  err);
    });
    return Result::SUCCESS;
}

Result SignalViewImplNativeCuda::presentSubmit() {
    return present();
}

Result SignalViewImplNativeCuda::computeInitialize() {
    if (lineplotEnabled) {
        JST_CHECK(createKernel(kLineplotKernelName, kLineplotKernelSource));
        lineplotKernelCreated = true;
    }
    if (waterfallEnabled) {
        JST_CHECK(createKernel(kLineplotWaterfallKernelName,
                               kLineplotWaterfallKernelSource));
        waterfallKernelCreated = true;
    }
    return Result::SUCCESS;
}

Result SignalViewImplNativeCuda::computeSubmit(const cudaStream_t& stream) {
    if (inputSampleSize == 0 || numberOfBatches == 0) {
        return Result::SUCCESS;
    }

    const auto* inputBase = static_cast<const std::uint8_t*>(input.buffer().data());
    if (!inputBase) {
        JST_ERROR("[MODULE_SIGNAL_VIEW_NATIVE_CUDA] Missing input buffer.");
        return Result::ERROR;
    }

    const void* inputData = inputBase + input.offsetBytes();
    void* inputArgument = const_cast<void*>(inputData);
    const Extent3D<U64> block = {kThreadsPerBlock, 1, 1};
    bool updateMaxHold = false;
    if (lineplotEnabled) {
        void* signalData = signalPoints.buffer().data();
        void* averageData = averagingBuffer.buffer().data();
        void* maxHoldData = maxHoldPoints.buffer().data();
        if (!signalData || !averageData || !maxHoldData) {
            JST_ERROR("[MODULE_SIGNAL_VIEW_NATIVE_CUDA] Missing lineplot "
                      "state buffer.");
            return Result::ERROR;
        }

        U64 averagingValue = averaging;
        updateMaxHold =
            maxHold && detail::LineplotMaxHoldReady(maxHoldWarmupBlocks, averaging);
        U32 maxHoldEnabled = updateMaxHold ? 1 : 0;
        void* arguments[] = {
            &inputArgument,
            &signalData,
            &averageData,
            &maxHoldData,
            &numberOfElements,
            &numberOfBatches,
            &inputBatchStride,
            &inputSampleStride,
            &decimation,
            &normalizationFactor,
            &averagingValue,
            &maxHoldEnabled,
        };
        const Extent3D<U64> grid = {
            lineplotGridSize,
            1,
            1,
        };
        JST_CHECK(scheduleKernel(kLineplotKernelName, stream, grid, block, arguments));
    }

    if (waterfallEnabled) {
        void* waterfallData = waterfallBins.buffer().data();
        if (!waterfallData) {
            JST_ERROR("[MODULE_SIGNAL_VIEW_NATIVE_CUDA] Missing waterfall "
                      "state buffer.");
            return Result::ERROR;
        }

        auto plan = PlanWaterfallWrite(waterfallHistory.writeIndex,
                                       numberOfBatches,
                                       waterfallHeight);
        void* waterfallArguments[] = {
            &inputArgument,
            &waterfallData,
            &inputSampleSize,
            &inputSampleStride,
            &inputBatchStride,
            &plan.rowCount,
            &waterfallHeight,
            &plan.sourceRow,
            &plan.destinationRow,
        };
        const Extent3D<U64> waterfallGrid = {
            waterfallGridSize,
            1,
            1,
        };
        JST_CHECK(scheduleKernel(kLineplotWaterfallKernelName,
                                 stream,
                                 waterfallGrid,
                                 block,
                                 waterfallArguments));
        waterfallHistory.advance(numberOfBatches, waterfallHeight);
    }

    if (lineplotEnabled && maxHold && maxHoldWarmupBlocks < averaging) {
        ++maxHoldWarmupBlocks;
    }

    if (lineplotEnabled) {
        updateSignalPointsFlag = true;
        if (updateMaxHold) { updateHoldPointsFlag = true; }
    }
    return Result::SUCCESS;
}

Result SignalViewImplNativeCuda::computeDeinitialize() {
    Result result = Result::SUCCESS;
    if (lineplotKernelCreated &&
        destroyKernel(kLineplotKernelName) != Result::SUCCESS) {
        result = Result::ERROR;
    }
    if (waterfallKernelCreated &&
        destroyKernel(kLineplotWaterfallKernelName) != Result::SUCCESS) {
        result = Result::ERROR;
    }

    lineplotKernelCreated = false;
    waterfallKernelCreated = false;
    return result;
}

JST_REGISTER_MODULE(SignalViewImplNativeCuda, DeviceType::CUDA, RuntimeType::NATIVE, "generic");

}  // namespace Jetstream::Modules
