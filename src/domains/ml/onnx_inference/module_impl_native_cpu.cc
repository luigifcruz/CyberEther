#include <jetstream/runtime_context_native_cpu.hh>
#include <jetstream/scheduler_context.hh>
#include <jetstream/module_context.hh>
#include <jetstream/registry.hh>

#include "module_impl.hh"

namespace Jetstream::Modules {

struct OnnxInferenceImplNativeCpu : public OnnxInferenceImpl,
                                    public NativeCpuRuntimeContext,
                                    public Scheduler::Context {
 public:
    Result create() final;

    Result computeSubmit() override;
};

Result OnnxInferenceImplNativeCpu::create() {
    return OnnxInferenceImpl::create();
}

Result OnnxInferenceImplNativeCpu::computeSubmit() {
    return runInference();
}

JST_REGISTER_MODULE(OnnxInferenceImplNativeCpu, DeviceType::CPU, RuntimeType::NATIVE, "generic");

}  // namespace Jetstream::Modules
