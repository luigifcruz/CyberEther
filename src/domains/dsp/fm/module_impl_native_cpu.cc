#include <cmath>
#include <complex>

#include <jetstream/backend/devices/cpu/helpers.hh>
#include <jetstream/runtime_context_native_cpu.hh>
#include <jetstream/scheduler_context.hh>
#include <jetstream/module_context.hh>
#include <jetstream/registry.hh>

#include "module_impl.hh"

namespace Jetstream::Modules {

struct FmImplNativeCpu : public FmImpl,
                         public NativeCpuRuntimeContext,
                         public Scheduler::Context {
 public:
    Result create() override;
    Result computeSubmit() override;
};

Result FmImplNativeCpu::create() {
    JST_CHECK(FmImpl::create());

    if (input.dtype() != DataType::CF32) {
        JST_ERROR("[MODULE_FM_NATIVE_CPU] Input must be complex (CF32).");
        return Result::ERROR;
    }

    return Result::SUCCESS;
}

Result FmImplNativeCpu::computeSubmit() {
    const CF32* inputData = input.data<CF32>();
    F32* outputData = output.data<F32>();
    const U64 size = input.size();
    const F32 refCoeff = ref;

    // First sample has no previous sample, set to zero.
    if (size > 0) {
        outputData[0] = 0.0f;
    }

    // Quadrature demodulation.
    for (U64 n = 1; n < size; n++) {
        outputData[n] = std::arg(std::conj(inputData[n - 1]) * inputData[n]) * refCoeff;
    }

    return Result::SUCCESS;
}

JST_REGISTER_MODULE(FmImplNativeCpu, DeviceType::CPU, RuntimeType::NATIVE, "generic");

}  // namespace Jetstream::Modules
