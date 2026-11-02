#include <jetstream/runtime_context_native_cpu.hh>
#include <jetstream/scheduler_context.hh>
#include <jetstream/module_context.hh>
#include <jetstream/registry.hh>

#include "module_impl.hh"

namespace Jetstream::Modules {

struct WebsocketReceiverImplNativeCpu : public WebsocketReceiverImpl,
                                        public NativeCpuRuntimeContext,
                                        public Scheduler::Context {
 public:
    Result computeSubmit() override;
    Result hasPendingCompute() override;
};

Result WebsocketReceiverImplNativeCpu::hasPendingCompute() {
    if (errored) {
        return Result::ERROR;
    }

    if (circularBuffer.size() < buffer.sizeBytes()) {
        return Result::TIMEOUT;
    }

    return Result::SUCCESS;
}

Result WebsocketReceiverImplNativeCpu::computeSubmit() {
    if (errored) {
        return Result::ERROR;
    }

    if (circularBuffer.size() < buffer.sizeBytes()) {
        return Result::YIELD;
    }

    const auto popped = circularBuffer.pop(reinterpret_cast<I8*>(buffer.data()), buffer.sizeBytes());
    if (popped == Result::INCOMPLETE) {
        return Result::YIELD;
    }

    return popped;
}

JST_REGISTER_MODULE(WebsocketReceiverImplNativeCpu, DeviceType::CPU, RuntimeType::NATIVE, "generic");

}  // namespace Jetstream::Modules
