#include <jetstream/module_context.hh>
#include <jetstream/registry.hh>
#include <jetstream/runtime_context_native_cuda.hh>
#include <jetstream/scheduler_context.hh>

#include "module_impl.hh"

namespace Jetstream::Modules {

namespace {

constexpr U64 kThreadsPerBlock = 256;
constexpr const char* kDuplicateStridedKernelName = "duplicate_strided";
constexpr const char* kDuplicateStridedKernelSource = R"(
extern "C" __global__ void duplicate_strided(const unsigned char* input,
                                             unsigned char* output,
                                             unsigned long long elementCount,
                                             unsigned long long elementSize,
                                             unsigned long long offsetBytes,
                                             int rank,
                                             const unsigned long long* shape,
                                             const unsigned long long* stride) {
    const unsigned long long index =
        (static_cast<unsigned long long>(blockIdx.x) * blockDim.x) + threadIdx.x;
    if (index >= elementCount) {
        return;
    }

    unsigned long long remaining = index;
    unsigned long long sourceIndex = 0;
    for (int axis = rank - 1; axis >= 0; --axis) {
        const unsigned long long coordinate = remaining % shape[axis];
        remaining /= shape[axis];
        sourceIndex += coordinate * stride[axis];
    }

    const unsigned char* src = input + offsetBytes + (sourceIndex * elementSize);
    unsigned char* dst = output + (index * elementSize);
    for (unsigned long long byteIndex = 0; byteIndex < elementSize; ++byteIndex) {
        dst[byteIndex] = src[byteIndex];
    }
}
)";

}  // namespace

struct DuplicateImplNativeCuda : public DuplicateImpl,
                                 public NativeCudaRuntimeContext,
                                 public Scheduler::Context {
 public:
    Result create() final;
    Result destroy() override;

    Result computeSubmit(const cudaStream_t& stream) override;
    Result computeDeinitialize() override;

 private:
    bool requiresKernel = false;
    bool kernelCreated = false;
    Tensor shapeTensor;
    Tensor strideTensor;
};

Result DuplicateImplNativeCuda::create() {
    JST_CHECK(DuplicateImpl::create());

    requiresKernel = !(input.contiguous() && input.sizeBytes() == input.buffer().sizeBytes());

    if (!requiresKernel) {
        return Result::SUCCESS;
    }

    Buffer::Config metadataConfig{};
    metadataConfig.hostAccessible = true;

    const Shape metadataShape = {static_cast<U64>(input.rank())};
    JST_CHECK(shapeTensor.create(device(), DataType::U64, metadataShape, metadataConfig));
    JST_CHECK(strideTensor.create(device(), DataType::U64, metadataShape, metadataConfig));

    auto* shapeData = shapeTensor.data<U64>();
    auto* strideData = strideTensor.data<U64>();

    for (Index axis = 0; axis < input.rank(); ++axis) {
        shapeData[axis] = input.shape(axis);
        strideData[axis] = input.stride(axis);
    }

    JST_CHECK(createKernel(kDuplicateStridedKernelName, kDuplicateStridedKernelSource));
    kernelCreated = true;

    return Result::SUCCESS;
}

Result DuplicateImplNativeCuda::destroy() {
    JST_CHECK(computeDeinitialize());
    return DuplicateImpl::destroy();
}

Result DuplicateImplNativeCuda::computeSubmit(const cudaStream_t& stream) {
    if (!requiresKernel) {
        return staging.copyFrom(input, stream);
    }

    const void* inputData = input.buffer().data();
    void* stagingData = staging.buffer().data();

    void* inputArgument = const_cast<void*>(inputData);
    U64 elementCount = input.size();
    U64 elementSize = input.elementSize();
    U64 offsetBytes = input.offsetBytes();
    I32 rank = static_cast<I32>(input.rank());
    void* shapeData = shapeTensor.data();
    void* strideData = strideTensor.data();

    void* arguments[] = {
        &inputArgument,
        &stagingData,
        &elementCount,
        &elementSize,
        &offsetBytes,
        &rank,
        &shapeData,
        &strideData,
    };

    const Extent3D<U64> block = {kThreadsPerBlock, 1, 1};
    const Extent3D<U64> grid = {(elementCount + kThreadsPerBlock - 1) / kThreadsPerBlock, 1, 1};

    return scheduleKernel(kDuplicateStridedKernelName, stream, grid, block, arguments);
}

Result DuplicateImplNativeCuda::computeDeinitialize() {
    if (kernelCreated) {
        JST_CHECK(destroyKernel(kDuplicateStridedKernelName));
    }

    kernelCreated = false;
    requiresKernel = false;
    shapeTensor = {};
    strideTensor = {};

    return Result::SUCCESS;
}

JST_REGISTER_MODULE(DuplicateImplNativeCuda, DeviceType::CUDA, RuntimeType::NATIVE, "generic");

}  // namespace Jetstream::Modules
