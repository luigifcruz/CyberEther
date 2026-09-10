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
constexpr const char* kWaterfallAverageKernelName = "waterfall_average_update";
constexpr const char* kLineplotKernelSource = R"(
extern "C" __global__ void lineplot_update(const float* input,
                                           float* signalPoints,
                                           float* lineplotAveragingBuffer,
                                           float* maxHoldPoints,
                                           unsigned long long numberOfElements,
                                           unsigned long long numberOfBatches,
                                           unsigned long long inputBatchStride,
                                           unsigned long long inputElementStride,
                                           float normalizationFactor,
                                           unsigned long long lineplotAveraging,
                                           unsigned int maxHoldEnabled,
                                           unsigned int averagingInitialized) {
    const unsigned long long index =
        (static_cast<unsigned long long>(blockIdx.x) * blockDim.x) + threadIdx.x;
    if (index >= numberOfElements) {
        return;
    }

    float sum = 0.0f;
    for (unsigned long long batch = 0; batch < numberOfBatches; ++batch) {
        const float value = input[(batch * inputBatchStride) + (index * inputElementStride)];
        sum += isfinite(value) ? value : (value > 0.0f ? 1.0f : 0.0f);
    }

    float amplitude = (sum * normalizationFactor) - 1.0f;
    if (!isfinite(amplitude)) {
        amplitude = fminf(fmaxf(amplitude, -1.0f), 1.0f);
    }
    float average = amplitude;
    if (averagingInitialized != 0 && isfinite(lineplotAveragingBuffer[index])) {
        average = lineplotAveragingBuffer[index];
        average -= average / static_cast<float>(lineplotAveraging);
        average += amplitude / static_cast<float>(lineplotAveraging);
    }

    lineplotAveragingBuffer[index] = average;
    const float displayed = tanhf(2.0f * average);
    signalPoints[(index * 2) + 1] = displayed;

    if (maxHoldEnabled != 0) {
        float& maxValue = maxHoldPoints[(index * 2) + 1];
        if (displayed > maxValue) {
            maxValue = displayed;
        }
    }
}
)";

constexpr const char* kLineplotWaterfallKernelSource = R"(
extern "C" __global__ void
lineplot_waterfall_update(const float* input,
                          float* waterfallBins,
                          unsigned long long elementCount,
                          unsigned long long inputElementStride,
                          unsigned long long inputBatchStride,
                          unsigned long long retainedBatches,
                          unsigned long long height,
                          unsigned long long sourceRow,
                          unsigned long long destinationRow) {
    const unsigned long long index =
        (static_cast<unsigned long long>(blockIdx.x) * blockDim.x) + threadIdx.x;
    const unsigned long long workItemCount = retainedBatches * elementCount;
    if (index >= workItemCount) {
        return;
    }

    const unsigned long long retainedBatch = index / elementCount;
    const unsigned long long element = index % elementCount;
    const unsigned long long sourceBatch = sourceRow + retainedBatch;
    const unsigned long long destinationBatch =
        (destinationRow + retainedBatch) % height;

    const float value = input[(sourceBatch * inputBatchStride) + (element * inputElementStride)];
    waterfallBins[(destinationBatch * elementCount) + element] =
        isfinite(value) ? value : (value > 0.0f ? 1.0f : 0.0f);
}
)";

constexpr const char* kWaterfallAverageKernelSource = R"(
extern "C" __global__ void
waterfall_average_update(const float* input,
                         float* waterfallBins,
                         double* waterfallAveragingBuffer,
                         unsigned long long elementCount,
                         unsigned long long numberOfBatches,
                         unsigned long long inputElementStride,
                         unsigned long long inputBatchStride,
                         unsigned long long height,
                         unsigned long long sourceRow,
                         unsigned long long writeIndex,
                         unsigned long long waterfallAveraging,
                         unsigned long long pendingRows) {
    const unsigned long long element =
        (static_cast<unsigned long long>(blockIdx.x) * blockDim.x) + threadIdx.x;
    if (element >= elementCount) {
        return;
    }

    double sum = pendingRows != 0 ? waterfallAveragingBuffer[element] : 0.0;
    unsigned long long outputRow = 0;
    for (unsigned long long batch = 0; batch < numberOfBatches; ++batch) {
        const float value = input[(batch * inputBatchStride) + (element * inputElementStride)];
        sum += static_cast<double>(isfinite(value) ? value : (value > 0.0f ? 1.0f : 0.0f));
        if (++pendingRows == waterfallAveraging) {
            if (outputRow >= sourceRow) {
                const unsigned long long destinationRow = (writeIndex + outputRow % height) % height;
                waterfallBins[(destinationRow * elementCount) + element] =
                    static_cast<float>(sum / static_cast<double>(waterfallAveraging));
            }
            pendingRows = 0;
            sum = 0.0;
            ++outputRow;
        }
    }
    waterfallAveragingBuffer[element] = sum;
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

 private:
    Tensor lineplotAveragingBuffer;
    Tensor waterfallAveragingBuffer;
    U64 lineplotGridSize = 0;
    U64 waterfallGridSize = 0;
    U64 validatedLineplotGridSize = 0;
    U64 validatedWaterfallGridSize = 0;
    bool lineplotKernelCreated = false;
    bool waterfallKernelCreated = false;
    bool waterfallAverageKernelCreated = false;
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
                                                validatedNumberOfElements,
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
        JST_CHECK(lineplotAveragingBuffer.create(device(), DataType::F32, {numberOfElements}));

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

    if (waterfallEnabled) {
        JST_CHECK(waterfallAveragingBuffer.create(device(), DataType::F64,
                                                  {numberOfElements}));
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
        JST_CHECK(createKernel(kWaterfallAverageKernelName,
                               kWaterfallAverageKernelSource));
        waterfallAverageKernelCreated = true;
    }
    return Result::SUCCESS;
}

Result SignalViewImplNativeCuda::computeSubmit(const cudaStream_t& stream) {
    if (numberOfElements == 0 || numberOfBatches == 0) {
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
        void* averageData = lineplotAveragingBuffer.buffer().data();
        void* maxHoldData = maxHoldPoints.buffer().data();
        if (!signalData || !averageData || !maxHoldData) {
            JST_ERROR("[MODULE_SIGNAL_VIEW_NATIVE_CUDA] Missing lineplot "
                      "state buffer.");
            return Result::ERROR;
        }

        updateMaxHold =
            maxHold && detail::LineplotMaxHoldReady(maxHoldWarmupBlocks, lineplotAveraging);
        U32 maxHoldEnabled = updateMaxHold ? 1 : 0;
        U32 averagingInitialized = lineplotAveragingInitialized ? 1 : 0;
        void* arguments[] = {
            &inputArgument,
            &signalData,
            &averageData,
            &maxHoldData,
            &numberOfElements,
            &numberOfBatches,
            &inputBatchStride,
            &inputElementStride,
            &normalizationFactor,
            &lineplotAveraging,
            &maxHoldEnabled,
            &averagingInitialized,
        };
        const Extent3D<U64> grid = {
            lineplotGridSize,
            1,
            1,
        };
        JST_CHECK(scheduleKernel(kLineplotKernelName, stream, grid, block, arguments));
        lineplotAveragingInitialized = true;
    }

    if (waterfallEnabled) {
        void* waterfallData = waterfallBins.buffer().data();
        if (!waterfallData) {
            JST_ERROR("[MODULE_SIGNAL_VIEW_NATIVE_CUDA] Missing waterfall "
                      "state buffer.");
            return Result::ERROR;
        }

        const auto averagePlan = PlanWaterfallAveraging(waterfallAveragingCount,
                                                        numberOfBatches, waterfallAveraging);
        auto plan = PlanWaterfallWrite(waterfallHistory.writeIndex,
                                       averagePlan.rowCount,
                                       waterfallHeight);
        if (waterfallAveraging > 1) {
            void* averageData = waterfallAveragingBuffer.buffer().data();
            if (!averageData) {
                JST_ERROR("[MODULE_SIGNAL_VIEW_NATIVE_CUDA] Missing waterfall averaging buffer.");
                return Result::ERROR;
            }
            void* waterfallArguments[] = {
                &inputArgument,
                &waterfallData,
                &averageData,
                &numberOfElements,
                &numberOfBatches,
                &inputElementStride,
                &inputBatchStride,
                &waterfallHeight,
                &plan.sourceRow,
                &waterfallHistory.writeIndex,
                &waterfallAveraging,
                &waterfallAveragingCount,
            };
            const Extent3D<U64> waterfallGrid = {
                numberOfElements / kThreadsPerBlock +
                    (numberOfElements % kThreadsPerBlock != 0),
                1,
                1,
            };
            JST_CHECK(scheduleKernel(kWaterfallAverageKernelName, stream,
                                     waterfallGrid, block, waterfallArguments));
        } else {
            void* waterfallArguments[] = {
                &inputArgument,
                &waterfallData,
                &numberOfElements,
                &inputElementStride,
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
        }
        waterfallAveragingCount = averagePlan.pendingRows;
        waterfallHistory.advance(averagePlan.rowCount, waterfallHeight);
    }

    if (lineplotEnabled && maxHold && maxHoldWarmupBlocks < lineplotAveraging) {
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
    if (waterfallAverageKernelCreated &&
        destroyKernel(kWaterfallAverageKernelName) != Result::SUCCESS) {
        result = Result::ERROR;
    }

    lineplotKernelCreated = false;
    waterfallKernelCreated = false;
    waterfallAverageKernelCreated = false;
    return result;
}

JST_REGISTER_MODULE(SignalViewImplNativeCuda, DeviceType::CUDA, RuntimeType::NATIVE, "generic");

}  // namespace Jetstream::Modules
