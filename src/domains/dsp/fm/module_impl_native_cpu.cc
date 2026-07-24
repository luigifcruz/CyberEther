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
    Result validate() override;
    Result computeSubmit() override;
};

Result FmImplNativeCpu::validate() {
    JST_CHECK(FmImpl::validate());

    if (!inputs().contains("signal")) {
        return Result::SUCCESS;
    }

    const Tensor& inputTensor = inputs().at("signal").tensor;
    if (!inputTensor.validShape() || inputTensor.size() == 0) {
        return Result::SUCCESS;
    }

    if (inputTensor.dtype() != DataType::CF32) {
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
