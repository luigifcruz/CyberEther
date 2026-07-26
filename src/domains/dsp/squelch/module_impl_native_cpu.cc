#include <algorithm>
#include <cmath>

#include <jetstream/runtime_context_native_cpu.hh>
#include <jetstream/scheduler_context.hh>
#include <jetstream/module_context.hh>
#include <jetstream/registry.hh>

#include "module_impl.hh"

namespace Jetstream::Modules {

struct SquelchImplNativeCpu : public SquelchImpl,
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

Result SquelchImplNativeCpu::validate() {
    JST_CHECK(SquelchImpl::validate());

    if (!inputs().contains("signal")) {
        return Result::SUCCESS;
    }

    const Tensor& inputTensor = inputs().at("signal").tensor;
    if (!inputTensor.validShape() || inputTensor.size() == 0) {
        return Result::SUCCESS;
    }

    if (inputTensor.dtype() == DataType::CF32) {
        return Result::SUCCESS;
    }

    if (inputTensor.dtype() == DataType::F32) {
        return Result::SUCCESS;
    }

    JST_ERROR("[MODULE_SQUELCH_NATIVE_CPU] Unsupported data type '{}'.",
              inputTensor.dtype());
    return Result::ERROR;
}

Result SquelchImplNativeCpu::create() {
    JST_CHECK(SquelchImpl::create());

    if (input.dtype() == DataType::CF32) {
        kernel = [this]() { return kernelCF32(); };
    } else {
        kernel = [this]() { return kernelF32(); };
    }

    return Result::SUCCESS;
}

Result SquelchImplNativeCpu::computeSubmit() {
    return kernel();
}

Result SquelchImplNativeCpu::kernelCF32() {
    const CF32* in = input.data<CF32>();
    const U64 size = input.size();
    F32 peakAmplitude = 0.0f;

    for (U64 i = 0; i < size; i++) {
        peakAmplitude = std::max(peakAmplitude, std::abs(in[i]));
    }

    const bool shouldPass = peakAmplitude > threshold;
    passingState.publish(shouldPass);
    amplitudeState.publish(peakAmplitude);
    return shouldPass ? Result::SUCCESS : Result::SKIP;
}

Result SquelchImplNativeCpu::kernelF32() {
    const F32* in = input.data<F32>();
    const U64 size = input.size();
    F32 peakAmplitude = 0.0f;

    for (U64 i = 0; i < size; i++) {
        peakAmplitude = std::max(peakAmplitude, std::fabs(in[i]));
    }

    const bool shouldPass = peakAmplitude > threshold;
    passingState.publish(shouldPass);
    amplitudeState.publish(peakAmplitude);
    return shouldPass ? Result::SUCCESS : Result::SKIP;
}

JST_REGISTER_MODULE(SquelchImplNativeCpu, DeviceType::CPU, RuntimeType::NATIVE, "generic");

}  // namespace Jetstream::Modules
