#include <thread>

#include <jetstream/backend/devices/cpu/helpers.hh>
#include <jetstream/runtime_context_native_cpu.hh>
#include <jetstream/scheduler_context.hh>
#include <jetstream/module_context.hh>
#include <jetstream/registry.hh>

#include "module_impl.hh"

namespace Jetstream::Modules {

struct ThrottleImplNativeCpu : public ThrottleImpl,
                                public Runtime::Context,
                                public Scheduler::Context {
 public:
    Result computeSubmit() override;
};

Result ThrottleImplNativeCpu::computeSubmit() {
    auto now = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        now - lastExecutionTime);

    // Sleep until the configured interval has elapsed.
    if (elapsed < std::chrono::milliseconds(intervalMs)) {
        std::this_thread::sleep_for(std::chrono::milliseconds(intervalMs) - elapsed);
    }

    // Update the timestamp for the next delay interval.
    lastExecutionTime = std::chrono::steady_clock::now();

    return Result::SUCCESS;
}

JST_REGISTER_MODULE(ThrottleImplNativeCpu, DeviceType::CPU, RuntimeType::NATIVE, "generic");

}  // namespace Jetstream::Modules
