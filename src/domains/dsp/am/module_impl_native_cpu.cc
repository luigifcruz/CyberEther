#include <cmath>
#include <complex>

#include <jetstream/backend/devices/cpu/helpers.hh>
#include <jetstream/runtime_context_native_cpu.hh>
#include <jetstream/scheduler_context.hh>
#include <jetstream/module_context.hh>
#include <jetstream/registry.hh>

#include "module_impl.hh"

namespace Jetstream::Modules {

struct AmImplNativeCpu : public AmImpl,
                         public NativeCpuRuntimeContext,
                         public Scheduler::Context {
 public:
    Result create() override;
    Result computeSubmit() override;
};

Result AmImplNativeCpu::create() {
    JST_CHECK(AmImpl::create());

    if (input.dtype() != DataType::CF32) {
        JST_ERROR("[MODULE_AM_NATIVE_CPU] Input must be complex (CF32).");
        return Result::ERROR;
    }

    return Result::SUCCESS;
}

Result AmImplNativeCpu::computeSubmit() {
    const CF32* inputData = input.data<CF32>();
    F32* outputData = output.data<F32>();
    const U64 size = input.size();
    const F32 alpha = dcAlpha;

    F32 prevEnv = prevEnvelope;
    F32 prevOut = prevOutput;

    for (U64 n = 0; n < size; n++) {
        const F32 envelope = std::abs(inputData[n]);
        outputData[n] = envelope - prevEnv + alpha * prevOut;
        prevEnv = envelope;
        prevOut = outputData[n];
    }

    prevEnvelope = prevEnv;
    prevOutput = prevOut;

    return Result::SUCCESS;
}

JST_REGISTER_MODULE(AmImplNativeCpu, DeviceType::CPU, RuntimeType::NATIVE, "generic");

}  // namespace Jetstream::Modules
