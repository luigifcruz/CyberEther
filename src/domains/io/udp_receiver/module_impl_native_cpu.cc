#include <jetstream/runtime_context_native_cpu.hh>
#include <jetstream/scheduler_context.hh>
#include <jetstream/module_context.hh>
#include <jetstream/registry.hh>

#include "module_impl.hh"

namespace Jetstream::Modules {

struct UdpReceiverImplNativeCpu : public UdpReceiverImpl,
                                  public NativeCpuRuntimeContext,
                                  public Scheduler::Context {
 public:
    Result computeSubmit() override;
    Result hasPendingCompute() override;
};

Result UdpReceiverImplNativeCpu::hasPendingCompute() {
    if (errored) {
        return Result::ERROR;
    }

    if (circularBuffer.size() < buffer.sizeBytes()) {
        return Result::TIMEOUT;
    }

    return Result::SUCCESS;
}

Result UdpReceiverImplNativeCpu::computeSubmit() {
    if (errored) {
        return Result::ERROR;
    }

    if (circularBuffer.size() < buffer.sizeBytes()) {
        return Result::YIELD;
    }

    JST_CHECK(circularBuffer.pop(
        reinterpret_cast<I8*>(buffer.data()), buffer.sizeBytes()));

    return Result::SUCCESS;
}

JST_REGISTER_MODULE(UdpReceiverImplNativeCpu, DeviceType::CPU, RuntimeType::NATIVE, "generic");

}  // namespace Jetstream::Modules
