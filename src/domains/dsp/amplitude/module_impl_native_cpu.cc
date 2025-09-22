#include <cmath>

#include <jetstream/tools/automatic_iterator.hh>
#include <jetstream/backend/devices/cpu/helpers.hh>
#include <jetstream/runtime_context_native_cpu.hh>
#include <jetstream/scheduler_context.hh>
#include <jetstream/module_context.hh>
#include <jetstream/registry.hh>

#include "module_impl.hh"

namespace Jetstream::Modules {

struct AmplitudeImplNativeCpu : public AmplitudeImpl,
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

Result AmplitudeImplNativeCpu::create() {
    // Create parent.

    JST_CHECK(AmplitudeImpl::create());

    // Register compute kernel.

    if (input.dtype() == DataType::CF32) {
        kernel = [this]() { return kernelCF32(); };
        return Result::SUCCESS;
    }

    if (input.dtype() == DataType::F32) {
        kernel = [this]() { return kernelF32(); };
        return Result::SUCCESS;
    }

    JST_ERROR("[MODULE_AMPLITUDE_NATIVE_CPU] Unsupported input data type: {}.", input.dtype());
    return Result::ERROR;
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
            out = 20.0f * Backend::ApproxLog10(magnitude) + coeff;
        },
    input, output);
}

Result AmplitudeImplNativeCpu::kernelF32() {
    const F32 coeff = scalingCoeff;

    return AutomaticIterator<F32, F32>(
        [coeff](const auto& in, auto& out) {
            const F32 magnitude = std::fabs(in);
            out = 20.0f * Backend::ApproxLog10(magnitude) + coeff;
        },
    input, output);
}

JST_REGISTER_MODULE(AmplitudeImplNativeCpu, DeviceType::CPU, RuntimeType::NATIVE, "generic");

}  // namespace Jetstream::Modules
