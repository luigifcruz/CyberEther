#include <jetstream/module_context.hh>
#include <jetstream/registry.hh>
#include <jetstream/runtime_context_native_cuda.hh>
#include <jetstream/scheduler_context.hh>

#include <cstdint>

#include "module_impl.hh"

namespace Jetstream::Modules {

namespace {

constexpr U64 kThreadsPerBlock = 256;
constexpr const char* kWindowKernelName = "window_kernel";
constexpr const char* kWindowKernelSource = R"(
struct alignas(8) KernelComplex {
    float real;
    float imag;
};

extern "C" __global__ void window_kernel(KernelComplex* output,
                                           unsigned long long elementCount) {
    const unsigned long long index =
        (static_cast<unsigned long long>(blockIdx.x) * blockDim.x) + threadIdx.x;
    if (index >= elementCount) {
        return;
    }

    if (elementCount == 1ULL) {
        output[index] = {1.0f, 0.0f};
        return;
    }

    const double pi = 3.14159265358979323846;
    const double denominator = static_cast<double>(elementCount - 1ULL);
    const double tap =
        0.42 - 0.50 * cos(2.0 * pi * static_cast<double>(index) / denominator) +
        0.08 * cos(4.0 * pi * static_cast<double>(index) / denominator);
    output[index] = {static_cast<float>(tap), 0.0f};
}
)";

}  // namespace

struct WindowImplNativeCuda : public WindowImpl,
                              public NativeCudaRuntimeContext,
                              public Scheduler::Context {
 public:
    Result computeInitialize() override;
    Result computeSubmit(const cudaStream_t& stream) override;
    Result computeDeinitialize() override;

 private:
    bool kernelCreated = false;
};

Result WindowImplNativeCuda::computeInitialize() {
    JST_CHECK(createKernel(kWindowKernelName, kWindowKernelSource));
    kernelCreated = true;
    return Result::SUCCESS;
}

Result WindowImplNativeCuda::computeSubmit(const cudaStream_t& stream) {
    if (baked) {
        return Result::SUCCESS;
    }

    U64 elementCount = output.size();
    if (elementCount == 0) {
        return Result::SUCCESS;
    }

    auto* outputBase = static_cast<std::uint8_t*>(output.buffer().data());
    if (!outputBase) {
        JST_ERROR("[MODULE_WINDOW_NATIVE_CUDA] Missing output device buffer.");
        return Result::ERROR;
    }

    void* outputData = outputBase + output.offsetBytes();
    void* arguments[] = {&outputData, &elementCount};

    const Extent3D<U64> block = {kThreadsPerBlock, 1, 1};
    const Extent3D<U64> grid = {
        (elementCount + kThreadsPerBlock - 1) / kThreadsPerBlock,
        1,
        1,
    };
    JST_CHECK(scheduleKernel(kWindowKernelName, stream, grid, block, arguments));

    baked = true;
    return Result::SUCCESS;
}

Result WindowImplNativeCuda::computeDeinitialize() {
    if (kernelCreated) {
        JST_CHECK(destroyKernel(kWindowKernelName));
    }
    kernelCreated = false;
    baked = false;
    return Result::SUCCESS;
}

JST_REGISTER_MODULE(WindowImplNativeCuda, DeviceType::CUDA, RuntimeType::NATIVE, "generic");

}  // namespace Jetstream::Modules
