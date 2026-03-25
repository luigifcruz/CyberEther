#include <jetstream/runtime_context_native_cpu.hh>
#include <jetstream/scheduler_context.hh>
#include <jetstream/module_context.hh>
#include <jetstream/registry.hh>

#include "module_impl.hh"

namespace Jetstream::Modules {

struct ReshapeImplNativeCpu : public ReshapeImpl,
                              public NativeCpuRuntimeContext,
                              public Scheduler::Context {
 public:
    Result computeSubmit() override;
};

Result ReshapeImplNativeCpu::computeSubmit() {
    // Reshape is a view operation with no data movement.
    return Result::SUCCESS;
}

JST_REGISTER_MODULE(ReshapeImplNativeCpu, DeviceType::CPU, RuntimeType::NATIVE, "generic");

}  // namespace Jetstream::Modules
