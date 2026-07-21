#include <cstdint>
#include <limits>
#include <vector>

#include <jetstream/backend/devices/cuda/helpers.hh>
#include <jetstream/module_context.hh>
#include <jetstream/registry.hh>
#include <jetstream/runtime_context_native_cuda.hh>
#include <jetstream/scheduler_context.hh>

#include "module_impl.hh"

namespace Jetstream::Modules {

namespace {

constexpr U64 kThreadsPerBlock = 256;
constexpr const char* kLineplotKernelName = "lineplot_update";
constexpr const char* kLineplotKernelSource = R"(
extern "C" __global__ void lineplot_update(const float* input,
                                           float* signalPoints,
                                           float* averagingBuffer,
                                           unsigned long long numberOfElements,
                                           unsigned long long numberOfBatches,
                                           unsigned long long inputRowWidth,
                                           unsigned long long decimation,
                                           float normalizationFactor,
                                           unsigned long long averaging) {
    const unsigned long long index =
        (static_cast<unsigned long long>(blockIdx.x) * blockDim.x) + threadIdx.x;
    if (index >= numberOfElements) {
        return;
    }

    float sum = 0.0f;
    for (unsigned long long batch = 0; batch < numberOfBatches; ++batch) {
        sum += input[(batch * inputRowWidth) + (index * decimation)];
    }

    const float amplitude = fminf(fmaxf((sum * normalizationFactor) - 1.0f, -1.0f), 1.0f);
    float average = averagingBuffer[index];
    average -= average / static_cast<float>(averaging);
    average += amplitude / static_cast<float>(averaging);

    averagingBuffer[index] = average;
    signalPoints[(index * 2) + 1] = average;
}
)";

}  // namespace

struct LineplotImplNativeCuda : public LineplotImpl,
                                public NativeCudaRuntimeContext,
                                public Scheduler::Context {
 public:
    Result create() final;

    Result presentInitialize() override;
    Result presentSubmit() override;
    Result computeInitialize() override;
    Result computeSubmit(const cudaStream_t& stream) override;
    Result computeDeinitialize() override;

 private:
    Result readSignalPoint(U64 index, F32* point) override;

    Tensor averagingBuffer;
    bool kernelCreated = false;
};

Result LineplotImplNativeCuda::create() {
    JST_CHECK(LineplotImpl::create());

    if (input.dtype() != DataType::F32) {
        JST_ERROR("[MODULE_LINEPLOT_NATIVE_CUDA] Unsupported input data type: {}.", input.dtype());
        return Result::ERROR;
    }

    if (numberOfElements == 0 || numberOfBatches == 0 || averaging == 0 || decimation == 0) {
        JST_ERROR("[MODULE_LINEPLOT_NATIVE_CUDA] Invalid zero-sized lineplot state.");
        return Result::ERROR;
    }

    const U64 blockCount = (numberOfElements + kThreadsPerBlock - 1) / kThreadsPerBlock;
    if (blockCount > std::numeric_limits<U32>::max()) {
        JST_ERROR("[MODULE_LINEPLOT_NATIVE_CUDA] Lineplot size exceeds the CUDA grid limit.");
        return Result::ERROR;
    }

    JST_CHECK(averagingBuffer.create(device(), DataType::F32, {numberOfElements}));

    std::vector<F32> initialPoints(signalPoints.size(), 0.0f);
    for (U64 index = 0; index < numberOfElements; ++index) {
        initialPoints[index * 2] = index * 2.0f / (numberOfElements - 1) - 1.0f;
    }

    auto* signalBase = static_cast<std::uint8_t*>(signalPoints.buffer().data());
    if (!signalBase) {
        JST_ERROR("[MODULE_LINEPLOT_NATIVE_CUDA] Missing signal points device buffer.");
        return Result::ERROR;
    }

    JST_CUDA_CHECK(cudaMemcpy(signalBase + signalPoints.offsetBytes(),
                              initialPoints.data(),
                              signalPoints.sizeBytes(),
                              cudaMemcpyHostToDevice), [&] {
        JST_ERROR("[MODULE_LINEPLOT_NATIVE_CUDA] Failed to initialize signal points: {}.", err);
    });

    return Result::SUCCESS;
}

Result LineplotImplNativeCuda::presentInitialize() {
    return createPresent();
}

Result LineplotImplNativeCuda::presentSubmit() {
    return present();
}

Result LineplotImplNativeCuda::computeInitialize() {
    JST_CHECK(createKernel(kLineplotKernelName, kLineplotKernelSource));
    kernelCreated = true;
    return Result::SUCCESS;
}

Result LineplotImplNativeCuda::computeSubmit(const cudaStream_t& stream) {
    if (numberOfElements == 0 || numberOfBatches == 0) {
        return Result::SUCCESS;
    }

    const auto* inputBase = static_cast<const std::uint8_t*>(input.buffer().data());
    void* signalData = signalPoints.buffer().data();
    void* averageData = averagingBuffer.buffer().data();
    if (!inputBase || !signalData || !averageData) {
        JST_ERROR("[MODULE_LINEPLOT_NATIVE_CUDA] Missing input or state buffer.");
        return Result::ERROR;
    }

    const void* inputData = inputBase + input.offsetBytes();
    void* inputArgument = const_cast<void*>(inputData);
    U64 averagingValue = averaging;

    void* arguments[] = {
        &inputArgument,
        &signalData,
        &averageData,
        &numberOfElements,
        &numberOfBatches,
        &inputRowWidth,
        &decimation,
        &normalizationFactor,
        &averagingValue,
    };

    const Extent3D<U64> block = {kThreadsPerBlock, 1, 1};
    const Extent3D<U64> grid = {
        (numberOfElements + kThreadsPerBlock - 1) / kThreadsPerBlock,
        1,
        1,
    };
    JST_CHECK(scheduleKernel(kLineplotKernelName, stream, grid, block, arguments));

    updateSignalPointsFlag = true;
    return Result::SUCCESS;
}

Result LineplotImplNativeCuda::computeDeinitialize() {
    Result result = Result::SUCCESS;
    if (kernelCreated && destroyKernel(kLineplotKernelName) != Result::SUCCESS) {
        result = Result::ERROR;
    }

    kernelCreated = false;
    return result;
}

Result LineplotImplNativeCuda::readSignalPoint(const U64 index, F32* point) {
    const auto* signalBase = static_cast<const std::uint8_t*>(signalPoints.buffer().data());
    if (!signalBase) {
        JST_ERROR("[MODULE_LINEPLOT_NATIVE_CUDA] Missing signal points device buffer.");
        return Result::ERROR;
    }

    const auto* source = signalBase + signalPoints.offsetBytes() +
                         (index * 2 * sizeof(F32));
    JST_CUDA_CHECK(cudaMemcpy(point, source, 2 * sizeof(F32), cudaMemcpyDeviceToHost), [&] {
        JST_ERROR("[MODULE_LINEPLOT_NATIVE_CUDA] Failed to read cursor point: {}.", err);
    });

    return Result::SUCCESS;
}

JST_REGISTER_MODULE(LineplotImplNativeCuda, DeviceType::CUDA, RuntimeType::NATIVE, "generic");

}  // namespace Jetstream::Modules
