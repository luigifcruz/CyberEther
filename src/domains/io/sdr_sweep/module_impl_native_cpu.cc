#include <jetstream/memory/macros.hh>
#include <jetstream/runtime_context_native_cpu.hh>
#include <jetstream/scheduler_context.hh>
#include <jetstream/module_context.hh>
#include <jetstream/registry.hh>

#include "module_impl.hh"

namespace Jetstream::Modules {

struct SdrSweepImplNativeCpu : public SdrSweepImpl,
                               public NativeCpuRuntimeContext,
                               public Scheduler::Context {
 public:
    Result computeSubmit() override;
    Result hasPendingCompute() override;
};

Result SdrSweepImplNativeCpu::hasPendingCompute() {
    if (errored) {
        return Result::ERROR;
    }

    std::lock_guard lock(slotMutex);
    expireFrame(Clock::now());
    return slotFresh ? Result::SUCCESS : Result::TIMEOUT;
}

Result SdrSweepImplNativeCpu::computeSubmit() {
    if (errored) {
        return Result::ERROR;
    }
    return emitFrame();
}

JST_REGISTER_MODULE(SdrSweepImplNativeCpu, DeviceType::CPU, RuntimeType::NATIVE, "generic");

}  // namespace Jetstream::Modules
