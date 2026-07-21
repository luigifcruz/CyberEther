// Looks like Windows static build crashes if multitheading is enabled.
#define POCKETFFT_NO_MULTITHREADING
#include "pocketfft.hh"

#include <jetstream/runtime_context_native_cpu.hh>
#include <jetstream/scheduler_context.hh>
#include <jetstream/module_context.hh>
#include <jetstream/registry.hh>
#include <jetstream/tools/automatic_iterator.hh>

#include "module_impl.hh"

namespace Jetstream::Modules {

struct FftImplNativeCpu : public FftImpl,
                          public NativeCpuRuntimeContext,
                          public Scheduler::Context {
 public:
    Result create() final;

    Result computeSubmit() override;

 private:
    template<typename T>
    Result applyInversion();

    Result kernelC2C();
    Result kernelR2C();
    Result kernelR2R();

    std::function<Result()> kernel;
    Tensor staging;
    U64 axisInnerSize = 1;
    U64 axisLength = 1;

    pocketfft::shape_t shape;
    pocketfft::stride_t inputStride;
    pocketfft::stride_t outputStride;
    pocketfft::shape_t axes;
};

Result FftImplNativeCpu::create() {
    // Create parent.

    JST_CHECK(FftImpl::create());

    if (invert) {
        JST_CHECK(staging.create(input.device(), input.dtype(), input.shape()));
    }

    axisInnerSize = 1;
    for (Index axisIndex = resolvedAxis + 1; axisIndex < input.rank(); ++axisIndex) {
        axisInnerSize *= input.shape(axisIndex);
    }
    axisLength = input.shape(resolvedAxis);

    // Setup pocketfft configuration.

    shape.clear();
    inputStride.clear();
    outputStride.clear();
    axes.clear();

    const Tensor& transformInput = invert ? staging : input;
    for (Index i = 0; i < input.rank(); ++i) {
        shape.push_back(static_cast<std::size_t>(input.shape(i)));
        inputStride.push_back(static_cast<std::ptrdiff_t>(transformInput.stride(i) *
                                                          transformInput.elementSize()));
        outputStride.push_back(static_cast<std::ptrdiff_t>(output.stride(i) *
                                                           output.elementSize()));
    }

    axes.push_back(static_cast<std::size_t>(resolvedAxis));

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
    if (invert) {
        if (input.dtype() == DataType::CF32) {
            JST_CHECK(applyInversion<CF32>());
        } else if (input.dtype() == DataType::F32) {
            JST_CHECK(applyInversion<F32>());
        }
    }

    return kernel();
}

template<typename T>
Result FftImplNativeCpu::applyInversion() {
    U64 index = 0;
    const U64 innerSize = axisInnerSize;
    const U64 length = axisLength;

    return AutomaticIterator<const T, T>(
        [&index, innerSize, length](const auto& in, auto& out) {
            const U64 axisCoordinate = (index / innerSize) % length;
            out = (axisCoordinate & 1ULL) != 0 ? -in : in;
            ++index;
        },
        input,
        staging);
}

Result FftImplNativeCpu::kernelC2C() {
    const Tensor& transformInput = invert ? staging : input;
    pocketfft::c2c(shape,
                   inputStride,
                   outputStride,
                   axes,
                   forward,
                   transformInput.data<CF32>(),
                   output.data<CF32>(),
                   1.0f);

    return Result::SUCCESS;
}

Result FftImplNativeCpu::kernelR2C() {
    const Tensor& transformInput = invert ? staging : input;
    pocketfft::r2c(shape,
                   inputStride,
                   outputStride,
                   axes,
                   forward,
                   transformInput.data<F32>(),
                   output.data<CF32>(),
                   1.0f);

    return Result::SUCCESS;
}

Result FftImplNativeCpu::kernelR2R() {
    const Tensor& transformInput = invert ? staging : input;
    pocketfft::r2r_fftpack(shape,
                           inputStride,
                           outputStride,
                           axes,
                           forward,
                           forward,
                           transformInput.data<F32>(),
                           output.data<F32>(),
                           1.0f);

    return Result::SUCCESS;
}

JST_REGISTER_MODULE(FftImplNativeCpu, DeviceType::CPU, RuntimeType::NATIVE, "generic");

}  // namespace Jetstream::Modules
