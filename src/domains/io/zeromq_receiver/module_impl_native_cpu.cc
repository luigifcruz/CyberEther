#include <jetstream/runtime_context_native_cpu.hh>
#include <jetstream/scheduler_context.hh>
#include <jetstream/module_context.hh>
#include <jetstream/registry.hh>

#include "module_impl.hh"

namespace Jetstream::Modules {

struct ZeromqReceiverImplNativeCpu : public ZeromqReceiverImpl,
                                     public NativeCpuRuntimeContext,
                                     public Scheduler::Context {
 public:
    Result computeSubmit() override;
    Result hasPendingCompute() override;
};

Result ZeromqReceiverImplNativeCpu::hasPendingCompute() {
    if (errored) {
        return Result::ERROR;
    }

    if (circularBuffer.size() < buffer.sizeBytes()) {
        return Result::TIMEOUT;
    }

    return Result::SUCCESS;
}

Result ZeromqReceiverImplNativeCpu::computeSubmit() {
    if (errored) {
        return Result::ERROR;
    }

    if (circularBuffer.size() < buffer.sizeBytes()) {
        return Result::YIELD;
    }

    return popBuffer();
}

JST_REGISTER_MODULE(ZeromqReceiverImplNativeCpu, DeviceType::CPU, RuntimeType::NATIVE, "generic");

}  // namespace Jetstream::Modules
