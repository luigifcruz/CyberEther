#include <jetstream/runtime_context_native_cpu.hh>
#include <jetstream/scheduler_context.hh>
#include <jetstream/module_context.hh>
#include <jetstream/registry.hh>

#include "module_impl.hh"

namespace Jetstream::Modules {

struct TcpTransmitterImplNativeCpu : public TcpTransmitterImpl,
                                     public NativeCpuRuntimeContext,
                                     public Scheduler::Context {
 public:
    Result computeSubmit() override;
};

Result TcpTransmitterImplNativeCpu::computeSubmit() {
    const auto& input = inputs().at("buffer").tensor;
    updateThroughput(send(input.data(), input.sizeBytes()) * input.sizeBytes());

    return Result::SUCCESS;
}

JST_REGISTER_MODULE(TcpTransmitterImplNativeCpu, DeviceType::CPU, RuntimeType::NATIVE, "generic");

}  // namespace Jetstream::Modules
