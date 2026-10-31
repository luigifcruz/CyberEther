#include <jetstream/runtime_context_native_cpu.hh>
#include <jetstream/scheduler_context.hh>
#include <jetstream/module_context.hh>
#include <jetstream/registry.hh>

#include "module_impl.hh"

namespace Jetstream::Modules {

struct ZeromqTransmitterImplNativeCpu : public ZeromqTransmitterImpl,
                                        public NativeCpuRuntimeContext,
                                        public Scheduler::Context {
 public:
    Result computeSubmit() override;
};

Result ZeromqTransmitterImplNativeCpu::computeSubmit() {
    send(inputs().at("buffer").tensor);

    return Result::SUCCESS;
}

JST_REGISTER_MODULE(ZeromqTransmitterImplNativeCpu, DeviceType::CPU, RuntimeType::NATIVE, "generic");

}  // namespace Jetstream::Modules
