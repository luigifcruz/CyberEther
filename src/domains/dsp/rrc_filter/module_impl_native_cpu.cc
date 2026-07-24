#include <cstring>
#include <limits>

#include <jetstream/backend/devices/cpu/helpers.hh>
#include <jetstream/memory/macros.hh>
#include <jetstream/runtime_context_native_cpu.hh>
#include <jetstream/scheduler_context.hh>
#include <jetstream/module_context.hh>
#include <jetstream/registry.hh>

#include "module_impl.hh"

namespace Jetstream::Modules {

struct RrcFilterImplNativeCpu : public RrcFilterImpl,
                                public NativeCpuRuntimeContext,
                                public Scheduler::Context {
 public:
    Result validate() final;
    Result create() final;

    Result computeSubmit() override;

 private:
    Result kernelCF32();
    Result kernelF32();

    std::function<Result()> kernel;
};

Result RrcFilterImplNativeCpu::validate() {
    JST_CHECK(RrcFilterImpl::validate());

    if (!inputs().contains("buffer")) {
        return Result::SUCCESS;
    }

    const Tensor& inputTensor = inputs().at("buffer").tensor;
    if (!inputTensor.validShape() || inputTensor.size() == 0) {
        return Result::SUCCESS;
    }

    if (inputTensor.dtype() != DataType::CF32 &&
        inputTensor.dtype() != DataType::F32) {
        JST_ERROR("[MODULE_RRC_FILTER_NATIVE_CPU] Unsupported input data type: {}.",
                  inputTensor.dtype());
        return Result::ERROR;
    }

    U64 coefficientBytes = 0;
    U64 historyBytes = 0;
    U64 alignedBytes = 0;
    if (!detail::CheckedMultiply(candidate()->taps,
                                 static_cast<U64>(sizeof(F32)),
                                 coefficientBytes) ||
        !detail::CheckedPageAlignedSize(coefficientBytes, alignedBytes) ||
        alignedBytes > std::numeric_limits<std::size_t>::max() ||
        !detail::CheckedMultiply(candidate()->taps,
                                 static_cast<U64>(DataTypeSize(inputTensor.dtype())),
                                 historyBytes) ||
        !detail::CheckedPageAlignedSize(historyBytes, alignedBytes) ||
        alignedBytes > std::numeric_limits<std::size_t>::max()) {
        JST_ERROR("[MODULE_RRC_FILTER_NATIVE_CPU] Tap buffers exceed the supported allocation range.");
        return Result::ERROR;
    }

    return Result::SUCCESS;
}

Result RrcFilterImplNativeCpu::create() {
    // Create parent.

    JST_CHECK(RrcFilterImpl::create());

    // Register compute kernel.

    if (input.dtype() == DataType::CF32) {
        kernel = [this]() { return kernelCF32(); };
    } else {
        kernel = [this]() { return kernelF32(); };
    }

    return Result::SUCCESS;
}

Result RrcFilterImplNativeCpu::computeSubmit() {
    return kernel();
}

Result RrcFilterImplNativeCpu::kernelCF32() {
    const U64 inputSize = input.size();
    const U64 numTaps = taps;
    const CF32* inPtr = input.data<CF32>();
    CF32* outPtr = output.data<CF32>();
    CF32* histPtr = history.data<CF32>();
    const F32* coeffPtr = coeffs.data<F32>();

    for (U64 n = 0; n < inputSize; ++n) {
        // Add current input sample to history.
        histPtr[historyIndex] = inPtr[n];

        // Compute filter output using convolution.
        CF32 outputSample{0.0f, 0.0f};

        for (U64 k = 0; k < numTaps; ++k) {
            const U64 histIdx = (historyIndex + numTaps - k) % numTaps;
            outputSample += histPtr[histIdx] * CF32(coeffPtr[k], 0.0f);
        }

        outPtr[n] = outputSample;

        // Update history index (circular buffer).
        historyIndex = (historyIndex + 1) % numTaps;
    }

    return Result::SUCCESS;
}

Result RrcFilterImplNativeCpu::kernelF32() {
    const U64 inputSize = input.size();
    const U64 numTaps = taps;
    const F32* inPtr = input.data<F32>();
    F32* outPtr = output.data<F32>();
    F32* histPtr = history.data<F32>();
    const F32* coeffPtr = coeffs.data<F32>();

    for (U64 n = 0; n < inputSize; ++n) {
        // Add current input sample to history.
        histPtr[historyIndex] = inPtr[n];

        // Compute filter output using convolution.
        F32 outputSample = 0.0f;

        for (U64 k = 0; k < numTaps; ++k) {
            const U64 histIdx = (historyIndex + numTaps - k) % numTaps;
            outputSample += histPtr[histIdx] * coeffPtr[k];
        }

        outPtr[n] = outputSample;

        // Update history index (circular buffer).
        historyIndex = (historyIndex + 1) % numTaps;
    }

    return Result::SUCCESS;
}

JST_REGISTER_MODULE(RrcFilterImplNativeCpu, DeviceType::CPU, RuntimeType::NATIVE, "generic");

}  // namespace Jetstream::Modules
