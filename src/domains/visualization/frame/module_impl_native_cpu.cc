#include <jetstream/runtime_context_native_cpu.hh>
#include <jetstream/scheduler_context.hh>
#include <jetstream/module_context.hh>
#include <jetstream/registry.hh>

#include "module_impl.hh"

namespace Jetstream::Modules {

struct FrameImplNativeCpu : public FrameImpl,
                            public NativeCpuRuntimeContext,
                            public Scheduler::Context {
 public:
    Result create() final;

    Result presentInitialize() override;
    Result presentSubmit() override;
    Result computeSubmit() override;
};

Result FrameImplNativeCpu::create() {
    JST_CHECK(FrameImpl::create());

    if (input.dtype() != DataType::F32) {
        JST_ERROR("[MODULE_FRAME_NATIVE_CPU] Unsupported input data type: {}.", input.dtype());
        return Result::ERROR;
    }

    return Result::SUCCESS;
}

Result FrameImplNativeCpu::presentInitialize() {
    return createPresent();
}

Result FrameImplNativeCpu::presentSubmit() {
    return present();
}

Result FrameImplNativeCpu::computeSubmit() {
    return Result::SUCCESS;
}

JST_REGISTER_MODULE(FrameImplNativeCpu, DeviceType::CPU, RuntimeType::NATIVE, "generic");

}  // namespace Jetstream::Modules
