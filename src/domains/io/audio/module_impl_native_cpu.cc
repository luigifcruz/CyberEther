#include <jetstream/runtime_context_native_cpu.hh>
#include <jetstream/scheduler_context.hh>
#include <jetstream/module_context.hh>
#include <jetstream/registry.hh>

#include "module_impl.hh"

namespace Jetstream::Modules {

struct AudioImplNativeCpu : public AudioImpl,
                            public NativeCpuRuntimeContext,
                            public Scheduler::Context {
 public:
    Result create() final;

    Result computeSubmit() override;
};

Result AudioImplNativeCpu::create() {
    JST_CHECK(AudioImpl::create());

    if (inputs().at("buffer").tensor.dtype() != DataType::F32) {
        JST_ERROR("[MODULE_AUDIO_NATIVE_CPU] Input buffer must be F32.");
        return Result::ERROR;
    }

    return Result::SUCCESS;
}

Result AudioImplNativeCpu::computeSubmit() {
    return resample();
}

JST_REGISTER_MODULE(AudioImplNativeCpu, DeviceType::CPU, RuntimeType::NATIVE, "generic");

}  // namespace Jetstream::Modules
