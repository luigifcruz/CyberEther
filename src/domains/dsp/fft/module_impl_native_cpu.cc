// Looks like Windows static build crashes if multitheading is enabled.
#define POCKETFFT_NO_MULTITHREADING
#include "pocketfft.hh"

#include <cstddef>
#include <limits>

#include <jetstream/memory/macros.hh>
#include <jetstream/runtime_context_native_cpu.hh>
#include <jetstream/scheduler_context.hh>
#include <jetstream/module_context.hh>
#include <jetstream/registry.hh>
#include <jetstream/tools/numeric.hh>

#include "module_impl.hh"

namespace Jetstream::Modules {

struct FftImplNativeCpu : public FftImpl,
                          public NativeCpuRuntimeContext,
                          public Scheduler::Context {
 public:
    Result validate() final;
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

Result FftImplNativeCpu::validate() {
    JST_CHECK(FftImpl::validate());

    if (!inputs().contains("signal")) {
        return Result::SUCCESS;
    }

    const Tensor& inputTensor = inputs().at("signal").tensor;
    if (!inputTensor.validShape() || inputTensor.size() == 0) {
        return Result::SUCCESS;
    }

    if (inputTensor.dtype() != DataType::F32 &&
        inputTensor.dtype() != DataType::CF32) {
        JST_ERROR("[MODULE_FFT_NATIVE_CPU] Unsupported input data type: {}.",
                  inputTensor.dtype());
        return Result::ERROR;
    }

    if (validatedOutputElementCount > std::numeric_limits<std::size_t>::max() ||
        validatedOutputSizeBytes >
            static_cast<U64>(std::numeric_limits<std::ptrdiff_t>::max())) {
        JST_ERROR("[MODULE_FFT_NATIVE_CPU] Transform dimensions exceed pocketfft limits.");
        return Result::ERROR;
    }

    U64 alignedOutputSize = 0;
    if (!detail::CheckedPageAlignedSize(validatedOutputSizeBytes,
                                        alignedOutputSize) ||
        alignedOutputSize > std::numeric_limits<std::size_t>::max()) {
        JST_ERROR("[MODULE_FFT_NATIVE_CPU] Output allocation size is too large.");
        return Result::ERROR;
    }

    for (Index axis = 0; axis < inputTensor.rank(); ++axis) {
        U64 strideBytes = 0;
        if (!detail::CheckedMultiply(inputTensor.stride(axis),
                                     inputTensor.elementSize(),
                                     strideBytes) ||
            strideBytes >
                static_cast<U64>(std::numeric_limits<std::ptrdiff_t>::max())) {
            JST_ERROR("[MODULE_FFT_NATIVE_CPU] Input strides exceed pocketfft limits.");
            return Result::ERROR;
        }
    }

    return Result::SUCCESS;
}

Result FftImplNativeCpu::create() {
    // Create parent.

    JST_CHECK(FftImpl::create());

    // Setup pocketfft configuration.

    shape.clear();
    inputStride.clear();
    outputStride.clear();
    axes.clear();

    for (Index i = 0; i < input.rank(); ++i) {
        shape.push_back(static_cast<std::size_t>(input.shape(i)));
        inputStride.push_back(static_cast<std::ptrdiff_t>(input.stride(i) *
                                                          input.elementSize()));
        outputStride.push_back(static_cast<std::ptrdiff_t>(output.stride(i) *
                                                           output.elementSize()));
    }

    axes.push_back(static_cast<std::size_t>(resolvedAxis));

    // Register compute kernel.

    if (input.dtype() == DataType::CF32 && output.dtype() == DataType::CF32) {
        kernel = [this]() { return kernelC2C(); };
    } else if (input.dtype() == DataType::F32 && output.dtype() == DataType::CF32) {
        kernel = [this]() { return kernelR2C(); };
    } else {
        kernel = [this]() { return kernelR2R(); };
    }

    return Result::SUCCESS;
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
                           forward,
                           forward,
                           input.data<F32>(),
                           output.data<F32>(),
                           1.0f);

    return Result::SUCCESS;
}

JST_REGISTER_MODULE(FftImplNativeCpu, DeviceType::CPU, RuntimeType::NATIVE, "generic");

}  // namespace Jetstream::Modules
