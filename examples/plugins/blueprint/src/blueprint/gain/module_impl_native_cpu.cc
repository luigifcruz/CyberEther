#include <functional>

#include <jetstream/module_context.hh>
#include <jetstream/registry.hh>
#include <jetstream/runtime_context_native_cpu.hh>
#include <jetstream/scheduler_context.hh>
#include <jetstream/tools/automatic_iterator.hh>

#include "module_impl.hh"

namespace Jetstream::Modules {

struct BlueprintGainImplNativeCpu : public BlueprintGainImpl,
                                    public NativeCpuRuntimeContext,
                                    public Scheduler::Context {
  public:
    Result create() final;

    Result computeSubmit() override;

  private:
    Result kernelF32();
    Result kernelCF32();

    std::function<Result()> kernel;
};

Result BlueprintGainImplNativeCpu::create() {
    JST_CHECK(BlueprintGainImpl::create());

    if (input.dtype() == DataType::F32 && output.dtype() == DataType::F32) {
        kernel = [this]() { return kernelF32(); };
        return Result::SUCCESS;
    }

    if (input.dtype() == DataType::CF32 && output.dtype() == DataType::CF32) {
        kernel = [this]() { return kernelCF32(); };
        return Result::SUCCESS;
    }

    JST_ERROR("[BLUEPRINT_GAIN_NATIVE_CPU] Unsupported data type '{}'.", input.dtype());
    return Result::ERROR;
}

Result BlueprintGainImplNativeCpu::computeSubmit() {
    return kernel();
}

Result BlueprintGainImplNativeCpu::kernelF32() {
    const F32 scale = gain;

    return AutomaticIterator<F32, F32>(
        [scale](const auto& in, auto& out) {
            out = in * scale;
        },
    input, output);
}

Result BlueprintGainImplNativeCpu::kernelCF32() {
    const F32 scale = gain;

    return AutomaticIterator<CF32, CF32>(
        [scale](const auto& in, auto& out) {
            out = in * scale;
        },
    input, output);
}

JST_REGISTER_MODULE(BlueprintGainImplNativeCpu, DeviceType::CPU, RuntimeType::NATIVE, "generic");

}  // namespace Jetstream::Modules
