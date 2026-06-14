#include <jetstream/runtime_context_native_cpu.hh>
#include <jetstream/scheduler_context.hh>
#include <jetstream/module_context.hh>
#include <jetstream/registry.hh>

#include "module_impl.hh"

namespace Jetstream::Modules {

struct InferImplNativeCpu : public InferImpl,
                            public NativeCpuRuntimeContext,
                            public Scheduler::Context {
 public:
    Result create() final;

    Result computeSubmit() override;
};

Result InferImplNativeCpu::create() {
    return InferImpl::create();
}

Result InferImplNativeCpu::computeSubmit() {
    return runInference();
}

JST_REGISTER_MODULE(InferImplNativeCpu, DeviceType::CPU, RuntimeType::NATIVE, "generic");

}  // namespace Jetstream::Modules
