#include <cmath>
#include <limits>

#include <jetstream/tools/automatic_iterator.hh>
#include <jetstream/backend/devices/cpu/helpers.hh>
#include <jetstream/runtime_context_native_cpu.hh>
#include <jetstream/scheduler_context.hh>
#include <jetstream/module_context.hh>
#include <jetstream/registry.hh>

#include "module_impl.hh"

namespace Jetstream::Modules {

struct AmplitudeImplNativeCpu : public AmplitudeImpl,
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

Result AmplitudeImplNativeCpu::validate() {
    JST_CHECK(AmplitudeImpl::validate());

    if (!inputs().contains("signal")) {
        return Result::SUCCESS;
    }

    const Tensor& inputTensor = inputs().at("signal").tensor;
    if (!inputTensor.validShape() || inputTensor.size() == 0) {
        return Result::SUCCESS;
    }

    if (inputTensor.dtype() != DataType::F32 &&
        inputTensor.dtype() != DataType::CF32) {
        JST_ERROR("[MODULE_AMPLITUDE_NATIVE_CPU] Unsupported input data type: {}.",
                  inputTensor.dtype());
        return Result::ERROR;
    }

    return Result::SUCCESS;
}

Result AmplitudeImplNativeCpu::create() {
    // Create parent.

    JST_CHECK(AmplitudeImpl::create());

    // Register compute kernel.

    if (input.dtype() == DataType::CF32) {
        kernel = [this]() { return kernelCF32(); };
    } else {
        kernel = [this]() { return kernelF32(); };
    }

    return Result::SUCCESS;
}

Result AmplitudeImplNativeCpu::computeSubmit() {
    return kernel();
}

Result AmplitudeImplNativeCpu::kernelCF32() {
    const F32 coeff = scalingCoeff;

    return AutomaticIterator<CF32, F32>(
        [coeff](const auto& in, auto& out) {
            const F32 real = in.real();
            const F32 imag = in.imag();
            const F32 magnitude = std::sqrt((real * real) + (imag * imag));
            out = magnitude == 0.0f
                      ? -std::numeric_limits<F32>::infinity()
                      : 20.0f * Backend::ApproxLog10(magnitude) + coeff;
        },
    input, output);
}

Result AmplitudeImplNativeCpu::kernelF32() {
    const F32 coeff = scalingCoeff;

    return AutomaticIterator<F32, F32>(
        [coeff](const auto& in, auto& out) {
            const F32 magnitude = std::fabs(in);
            out = magnitude == 0.0f
                      ? -std::numeric_limits<F32>::infinity()
                      : 20.0f * Backend::ApproxLog10(magnitude) + coeff;
        },
    input, output);
}

JST_REGISTER_MODULE(AmplitudeImplNativeCpu, DeviceType::CPU, RuntimeType::NATIVE, "generic");

}  // namespace Jetstream::Modules
