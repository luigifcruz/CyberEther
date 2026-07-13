#include <algorithm>
#include <cstdint>
#include <limits>

#include <jetstream/module_context.hh>
#include <jetstream/registry.hh>
#include <jetstream/runtime_context_native_cuda.hh>
#include <jetstream/scheduler_context.hh>

#include "module_impl.hh"

namespace Jetstream::Modules {

namespace {

constexpr U64 kThreadsPerBlock = 256;
constexpr const char* kWaterfallKernelName = "waterfall_update";
constexpr const char* kWaterfallKernelSource = R"(
extern "C" __global__ void waterfall_update(const float* input,
                                            float* frequencyBins,
                                            unsigned long long numberOfElements,
                                            unsigned long long retainedBatches,
                                            unsigned long long height,
                                            unsigned long long sourceRow,
                                            unsigned long long destinationRow) {
    const unsigned long long index =
        (static_cast<unsigned long long>(blockIdx.x) * blockDim.x) + threadIdx.x;
    const unsigned long long elementCount = retainedBatches * numberOfElements;
    if (index >= elementCount) {
        return;
    }

    const unsigned long long retainedBatch = index / numberOfElements;
    const unsigned long long column = index % numberOfElements;
    const unsigned long long sourceBatch = sourceRow + retainedBatch;
    const unsigned long long destinationBatch = (destinationRow + retainedBatch) % height;

    frequencyBins[(destinationBatch * numberOfElements) + column] =
        input[(sourceBatch * numberOfElements) + column];
}
)";

}  // namespace

struct WaterfallImplNativeCuda : public WaterfallImpl,
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
    bool kernelCreated = false;
};

Result WaterfallImplNativeCuda::create() {
    JST_CHECK(WaterfallImpl::create());

    if (input.dtype() != DataType::F32) {
        JST_ERROR("[MODULE_WATERFALL_NATIVE_CUDA] Unsupported input data type: {}.", input.dtype());
        return Result::ERROR;
    }

    if (numberOfElements == 0 || numberOfBatches == 0 || height == 0) {
        JST_ERROR("[MODULE_WATERFALL_NATIVE_CUDA] Invalid zero-sized waterfall state.");
        return Result::ERROR;
    }

    const U64 retainedBatches = std::min(numberOfBatches, height);
    const U64 elementCount = retainedBatches * numberOfElements;
    const U64 blockCount = (elementCount + kThreadsPerBlock - 1) / kThreadsPerBlock;
    if (blockCount > std::numeric_limits<U32>::max()) {
        JST_ERROR("[MODULE_WATERFALL_NATIVE_CUDA] Waterfall size exceeds the CUDA grid limit.");
        return Result::ERROR;
    }

    return Result::SUCCESS;
}

Result WaterfallImplNativeCuda::presentInitialize() {
    return createPresent();
}

Result WaterfallImplNativeCuda::presentSubmit() {
    return present();
}

Result WaterfallImplNativeCuda::computeInitialize() {
    JST_CHECK(createKernel(kWaterfallKernelName, kWaterfallKernelSource));
    kernelCreated = true;
    return Result::SUCCESS;
}

Result WaterfallImplNativeCuda::computeSubmit(const cudaStream_t& stream) {
    if (numberOfElements == 0 || numberOfBatches == 0) {
        return Result::SUCCESS;
    }

    const auto* inputBase = static_cast<const std::uint8_t*>(input.buffer().data());
    void* frequencyData = frequencyBins.buffer().data();
    if (!inputBase || !frequencyData) {
        JST_ERROR("[MODULE_WATERFALL_NATIVE_CUDA] Missing input or frequency bins buffer.");
        return Result::ERROR;
    }

    const void* inputData = inputBase + input.offsetBytes();
    void* inputArgument = const_cast<void*>(inputData);
    auto plan = PlanWaterfallWrite(ringState.writeIndex,
                                   numberOfBatches,
                                   height);
    const U64 elementCount = plan.rowCount * numberOfElements;

    void* arguments[] = {
        &inputArgument,
        &frequencyData,
        &numberOfElements,
        &plan.rowCount,
        &height,
        &plan.sourceRow,
        &plan.destinationRow,
    };

    const Extent3D<U64> block = {kThreadsPerBlock, 1, 1};
    const Extent3D<U64> grid = {
        (elementCount + kThreadsPerBlock - 1) / kThreadsPerBlock,
        1,
        1,
    };
    JST_CHECK(scheduleKernel(kWaterfallKernelName, stream, grid, block, arguments));

    ringState.advance(numberOfBatches, height);
    return Result::SUCCESS;
}

Result WaterfallImplNativeCuda::computeDeinitialize() {
    Result result = Result::SUCCESS;
    if (kernelCreated && destroyKernel(kWaterfallKernelName) != Result::SUCCESS) {
        result = Result::ERROR;
    }

    kernelCreated = false;
    return result;
}

JST_REGISTER_MODULE(WaterfallImplNativeCuda, DeviceType::CUDA, RuntimeType::NATIVE, "generic");

}  // namespace Jetstream::Modules
