#include <thread>

#include <jetstream/backend/devices/cpu/helpers.hh>
#include <jetstream/runtime_context_native_cpu.hh>
#include <jetstream/scheduler_context.hh>
#include <jetstream/module_context.hh>
#include <jetstream/registry.hh>

#include "module_impl.hh"

namespace Jetstream::Modules {

struct ThrottleImplNativeCpu : public ThrottleImpl,
                               public NativeCpuRuntimeContext,
                               public Scheduler::Context {
 public:
    Result computeSubmit() override;
};

Result ThrottleImplNativeCpu::computeSubmit() {
    if (std::chrono::steady_clock::now() < schedule.deadline) {
        std::this_thread::sleep_until(schedule.deadline);
    }

    schedule.advance(std::chrono::steady_clock::now());

    return Result::SUCCESS;
}

JST_REGISTER_MODULE(ThrottleImplNativeCpu, DeviceType::CPU, RuntimeType::NATIVE, "generic");

}  // namespace Jetstream::Modules
