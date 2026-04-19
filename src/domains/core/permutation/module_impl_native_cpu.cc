#include <jetstream/runtime_context_native_cpu.hh>
#include <jetstream/scheduler_context.hh>
#include <jetstream/module_context.hh>
#include <jetstream/registry.hh>

#include "module_impl.hh"

namespace Jetstream::Modules {

struct PermutationImplNativeCpu : public PermutationImpl,
                                  public NativeCpuRuntimeContext,
                                  public Scheduler::Context {
 public:
    Result create() final;
    Result computeSubmit() override;
};

Result PermutationImplNativeCpu::create() {
    JST_CHECK(PermutationImpl::create());

    return Result::SUCCESS;
}

Result PermutationImplNativeCpu::computeSubmit() {
    return Result::SUCCESS;
}

JST_REGISTER_MODULE(PermutationImplNativeCpu, DeviceType::CPU, RuntimeType::NATIVE, "generic");

}  // namespace Jetstream::Modules
