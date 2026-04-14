#include <jetstream/module_context.hh>
#include <jetstream/registry.hh>
#include <jetstream/runtime_context_native_cpu.hh>
#include <jetstream/scheduler_context.hh>

#include "module_impl.hh"

namespace Jetstream::Modules {

struct OnesTensorImplNativeCpu : public OnesTensorImpl,
                                 public NativeCpuRuntimeContext,
                                 public Scheduler::Context {
 public:
    Result computeSubmit() override;
};

Result OnesTensorImplNativeCpu::computeSubmit() {
    return Result::SUCCESS;
}

JST_REGISTER_MODULE(OnesTensorImplNativeCpu, DeviceType::CPU, RuntimeType::NATIVE, "generic");

}  // namespace Jetstream::Modules
