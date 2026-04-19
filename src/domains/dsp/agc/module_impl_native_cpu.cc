#include <cmath>

#include <jetstream/runtime_context_native_cpu.hh>
#include <jetstream/scheduler_context.hh>
#include <jetstream/module_context.hh>
#include <jetstream/registry.hh>

#include "module_impl.hh"

namespace Jetstream::Modules {

struct AgcImplNativeCpu : public AgcImpl,
                          public NativeCpuRuntimeContext,
                          public Scheduler::Context {
 public:
    Result create() final;

    Result computeSubmit() override;

 private:
    Result kernelCF32();
    Result kernelF32();

    std::function<Result()> kernel;
};

Result AgcImplNativeCpu::create() {
    // Create parent.

    JST_CHECK(AgcImpl::create());

    // Register compute kernel.

    if (input.dtype() == DataType::CF32) {
        kernel = [this]() { return kernelCF32(); };
        return Result::SUCCESS;
    }

    if (input.dtype() == DataType::F32) {
        kernel = [this]() { return kernelF32(); };
        return Result::SUCCESS;
    }

    JST_ERROR("[MODULE_AGC_NATIVE_CPU] Unsupported data type '{}'.", input.dtype());
    return Result::ERROR;
}

Result AgcImplNativeCpu::computeSubmit() {
    return kernel();
}

Result AgcImplNativeCpu::kernelCF32() {
    const CF32* in = input.data<CF32>();
    CF32* out = output.data<CF32>();
    const U64 size = input.size();

    constexpr F32 desiredLevel = 1.0f;

    // Find maximum absolute value.
    F32 currentMax = 0.0f;
    for (U64 i = 0; i < size; i++) {
        currentMax = std::max(currentMax, std::abs(in[i]));
    }

    // Calculate gain.
    const F32 gain = (currentMax != 0.0f) ? (desiredLevel / currentMax) : 1.0f;

    // Apply gain.
    for (U64 i = 0; i < size; i++) {
        out[i] = in[i] * gain;
    }

    return Result::SUCCESS;
}

Result AgcImplNativeCpu::kernelF32() {
    const F32* in = input.data<F32>();
    F32* out = output.data<F32>();
    const U64 size = input.size();

    constexpr F32 desiredLevel = 1.0f;

    // Find maximum absolute value.
    F32 currentMax = 0.0f;
    for (U64 i = 0; i < size; i++) {
        currentMax = std::max(currentMax, std::fabs(in[i]));
    }

    // Calculate gain.
    const F32 gain = (currentMax != 0.0f) ? (desiredLevel / currentMax) : 1.0f;

    // Apply gain.
    for (U64 i = 0; i < size; i++) {
        out[i] = in[i] * gain;
    }

    return Result::SUCCESS;
}

JST_REGISTER_MODULE(AgcImplNativeCpu, DeviceType::CPU, RuntimeType::NATIVE, "generic");

}  // namespace Jetstream::Modules
