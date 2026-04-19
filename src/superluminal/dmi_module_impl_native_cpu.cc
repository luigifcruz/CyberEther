#include <jetstream/runtime_context_native_cpu.hh>
#include <jetstream/scheduler_context.hh>
#include <jetstream/module_context.hh>
#include <jetstream/registry.hh>

#include "dmi_module_impl.hh"

namespace Jetstream::Modules {

struct DynamicTensorImportImplNativeCpu : public DynamicTensorImportImpl,
                                          public NativeCpuRuntimeContext,
                                          public Scheduler::Context {
 public:
    Result computeSubmit() override;
};

Result DynamicTensorImportImplNativeCpu::computeSubmit() {
    // No-op: data comes from external memory.
    return Result::SUCCESS;
}

JST_REGISTER_MODULE(DynamicTensorImportImplNativeCpu, DeviceType::CPU, RuntimeType::NATIVE, "generic");

}  // namespace Jetstream::Modules
