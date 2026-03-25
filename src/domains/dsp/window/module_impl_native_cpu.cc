#include <cmath>

#include <jetstream/backend/devices/cpu/helpers.hh>
#include <jetstream/runtime_context_native_cpu.hh>
#include <jetstream/scheduler_context.hh>
#include <jetstream/module_context.hh>
#include <jetstream/registry.hh>

#include "module_impl.hh"

namespace Jetstream::Modules {

struct WindowImplNativeCpu : public WindowImpl,
                             public NativeCpuRuntimeContext,
                             public Scheduler::Context {
 public:
    Result computeSubmit() override;
};

Result WindowImplNativeCpu::computeSubmit() {
    // Window only needs to be computed once.
    if (baked) {
        return Result::SUCCESS;
    }

    // Generate Blackman window.
    const U64 N = size;
    CF32* windowData = output.data<CF32>();

    for (U64 i = 0; i < N; i++) {
        const F64 tap = 0.42 - 0.50 * std::cos(2.0 * JST_PI * i / (N - 1)) +
                        0.08 * std::cos(4.0 * JST_PI * i / (N - 1));
        windowData[i] = CF32(static_cast<F32>(tap), 0.0f);
    }

    baked = true;

    return Result::SUCCESS;
}

JST_REGISTER_MODULE(WindowImplNativeCpu, DeviceType::CPU, RuntimeType::NATIVE, "generic");

}  // namespace Jetstream::Modules
