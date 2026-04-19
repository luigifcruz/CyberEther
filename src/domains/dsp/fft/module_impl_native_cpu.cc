// Looks like Windows static build crashes if multitheading is enabled.
#define POCKETFFT_NO_MULTITHREADING
#include "pocketfft.hh"

#include <jetstream/runtime_context_native_cpu.hh>
#include <jetstream/scheduler_context.hh>
#include <jetstream/module_context.hh>
#include <jetstream/registry.hh>

#include "module_impl.hh"

namespace Jetstream::Modules {

struct FftImplNativeCpu : public FftImpl,
                          public NativeCpuRuntimeContext,
                          public Scheduler::Context {
 public:
    Result create() final;

    Result computeSubmit() override;

 private:
    Result kernelC2C();
    Result kernelR2C();
    Result kernelR2R();

    std::function<Result()> kernel;

    pocketfft::shape_t shape;
    pocketfft::stride_t inputStride;
    pocketfft::stride_t outputStride;
    pocketfft::shape_t axes;
};

Result FftImplNativeCpu::create() {
    // Create parent.

    JST_CHECK(FftImpl::create());

    // Setup pocketfft configuration.

    shape.clear();
    inputStride.clear();
    outputStride.clear();
    axes.clear();

    for (U64 i = 0; i < input.rank(); ++i) {
        shape.push_back(static_cast<U32>(input.shape()[i]));
        inputStride.push_back(static_cast<U32>(input.stride()[i]) * input.elementSize());
        outputStride.push_back(static_cast<U32>(output.stride()[i]) * output.elementSize());
    }

    axes.push_back(output.rank() - 1);

    // Register compute kernel.

    if (input.dtype() == DataType::CF32 && output.dtype() == DataType::CF32) {
        kernel = [this]() { return kernelC2C(); };
        return Result::SUCCESS;
    }

    if (input.dtype() == DataType::F32 && output.dtype() == DataType::CF32) {
        kernel = [this]() { return kernelR2C(); };
        return Result::SUCCESS;
    }

    if (input.dtype() == DataType::F32 && output.dtype() == DataType::F32) {
        kernel = [this]() { return kernelR2R(); };
        return Result::SUCCESS;
    }

    JST_ERROR("[MODULE_FFT_NATIVE_CPU] Unsupported data type combination: {} -> {}.",
              input.dtype(), output.dtype());
    return Result::ERROR;
}

Result FftImplNativeCpu::computeSubmit() {
    return kernel();
}

Result FftImplNativeCpu::kernelC2C() {
    pocketfft::c2c(shape,
                   inputStride,
                   outputStride,
                   axes,
                   forward,
                   input.data<CF32>(),
                   output.data<CF32>(),
                   1.0f);

    return Result::SUCCESS;
}

Result FftImplNativeCpu::kernelR2C() {
    pocketfft::r2c(shape,
                   inputStride,
                   outputStride,
                   axes,
                   forward,
                   input.data<F32>(),
                   output.data<CF32>(),
                   1.0f);

    return Result::SUCCESS;
}

Result FftImplNativeCpu::kernelR2R() {
    pocketfft::r2r_fftpack(shape,
                           inputStride,
                           outputStride,
                           axes,
                           true,  // real2hermitian
                           forward,
                           input.data<F32>(),
                           output.data<F32>(),
                           1.0f);

    return Result::SUCCESS;
}

JST_REGISTER_MODULE(FftImplNativeCpu, DeviceType::CPU, RuntimeType::NATIVE, "generic");

}  // namespace Jetstream::Modules
