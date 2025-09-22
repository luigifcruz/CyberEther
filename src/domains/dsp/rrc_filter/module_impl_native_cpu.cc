#include <cstring>

#include <jetstream/backend/devices/cpu/helpers.hh>
#include <jetstream/runtime_context_native_cpu.hh>
#include <jetstream/scheduler_context.hh>
#include <jetstream/module_context.hh>
#include <jetstream/registry.hh>

#include "module_impl.hh"

namespace Jetstream::Modules {

struct RrcFilterImplNativeCpu : public RrcFilterImpl,
                                public Runtime::Context,
                                public Scheduler::Context {
 public:
    Result create() final;

    Result computeSubmit() override;

 private:
    Result kernelCF32();
    Result kernelF32();

    std::function<Result()> kernel;
};

Result RrcFilterImplNativeCpu::create() {
    // Create parent.

    JST_CHECK(RrcFilterImpl::create());

    // Register compute kernel.

    if (input.dtype() == DataType::CF32) {
        kernel = [this]() { return kernelCF32(); };
        return Result::SUCCESS;
    }

    if (input.dtype() == DataType::F32) {
        kernel = [this]() { return kernelF32(); };
        return Result::SUCCESS;
    }

    JST_ERROR("[MODULE_RRC_FILTER_NATIVE_CPU] Unsupported input "
              "data type: {}.", input.dtype());
    return Result::ERROR;
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
