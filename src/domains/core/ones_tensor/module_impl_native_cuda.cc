#include <jetstream/module_context.hh>
#include <jetstream/registry.hh>
#include <jetstream/runtime_context_native_cuda.hh>
#include <jetstream/scheduler_context.hh>

#include <cstdint>

#include "module_impl.hh"

namespace Jetstream::Modules {

namespace {

constexpr U64 kThreadsPerBlock = 256;
constexpr const char* kOnesTensorKernelName = "ones_tensor_kernel";
static_assert(sizeof(CF32) == 2 * sizeof(F32));
static_assert(sizeof(CF64) == 2 * sizeof(F64));

constexpr const char* kOnesTensorKernelSource = R"(
extern "C" __global__ void ones_tensor_kernel(unsigned char* output,
                                               unsigned long long elementCount,
                                               unsigned int dataType) {
    const unsigned long long index =
        (static_cast<unsigned long long>(blockIdx.x) * blockDim.x) + threadIdx.x;
    if (index >= elementCount) {
        return;
    }

    if (dataType == 0U) {
        reinterpret_cast<float*>(output)[index] = 1.0f;
    } else if (dataType == 1U) {
        float* complexOutput = reinterpret_cast<float*>(output) + (index * 2ULL);
        complexOutput[0] = 1.0f;
        complexOutput[1] = 0.0f;
    } else if (dataType == 2U) {
        reinterpret_cast<double*>(output)[index] = 1.0;
    } else {
        double* complexOutput = reinterpret_cast<double*>(output) + (index * 2ULL);
        complexOutput[0] = 1.0;
        complexOutput[1] = 0.0;
    }
}
)";

}  // namespace

struct OnesTensorImplNativeCuda : public OnesTensorImpl,
                                  public NativeCudaRuntimeContext,
                                  public Scheduler::Context {
 public:
    Result computeInitialize() override;
    Result computeSubmit(const cudaStream_t& stream) override;
    Result computeDeinitialize() override;

 private:
    bool kernelCreated = false;
};

Result OnesTensorImplNativeCuda::computeInitialize() {
    JST_CHECK(createKernel(kOnesTensorKernelName, kOnesTensorKernelSource));
    kernelCreated = true;

    return Result::SUCCESS;
}

Result OnesTensorImplNativeCuda::computeSubmit(const cudaStream_t& stream) {
    U32 dataType = 0;
    switch (output.dtype()) {
        case DataType::F32:
            dataType = 0;
            break;
        case DataType::CF32:
            dataType = 1;
            break;
        case DataType::F64:
            dataType = 2;
            break;
        case DataType::CF64:
            dataType = 3;
            break;
        default:
            JST_ERROR("[MODULE_ONES_TENSOR_NATIVE_CUDA] Unsupported data type '{}'.", output.dtype());
            return Result::ERROR;
    }

    U64 elementCount = output.size();
    auto* outputBase = static_cast<std::uint8_t*>(output.buffer().data());
    if (!outputBase) {
        JST_ERROR("[MODULE_ONES_TENSOR_NATIVE_CUDA] Missing output device buffer.");
        return Result::ERROR;
    }

    void* outputData = outputBase + output.offsetBytes();
    void* arguments[] = {&outputData, &elementCount, &dataType};

    const Extent3D<U64> block = {kThreadsPerBlock, 1, 1};
    const Extent3D<U64> grid = {
        (elementCount + kThreadsPerBlock - 1) / kThreadsPerBlock,
        1,
        1,
    };

    return scheduleKernel(kOnesTensorKernelName, stream, grid, block, arguments);
}

Result OnesTensorImplNativeCuda::computeDeinitialize() {
    if (kernelCreated) {
        JST_CHECK(destroyKernel(kOnesTensorKernelName));
    }
    kernelCreated = false;

    return Result::SUCCESS;
}

JST_REGISTER_MODULE(OnesTensorImplNativeCuda, DeviceType::CUDA, RuntimeType::NATIVE, "generic");

}  // namespace Jetstream::Modules
