#include <jetstream/module_context.hh>
#include <jetstream/registry.hh>
#include <jetstream/runtime_context_native_cpu.hh>
#include <jetstream/scheduler_context.hh>

#include "module_impl.hh"

namespace Jetstream::Modules {

struct SignalAttributesImplNativeCpu : public SignalAttributesImpl,
                                 public NativeCpuRuntimeContext,
                                 public Scheduler::Context {
    Result computeSubmit() override;
};

Result SignalAttributesImplNativeCpu::computeSubmit() {
    return Result::SUCCESS;
}

JST_REGISTER_MODULE(SignalAttributesImplNativeCpu, DeviceType::CPU, RuntimeType::NATIVE, "generic");

}  // namespace Jetstream::Modules
