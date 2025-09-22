#include <jetstream/runtime_context_native_cpu.hh>
#include <jetstream/scheduler_context.hh>
#include <jetstream/module_context.hh>
#include <jetstream/registry.hh>

#include "module_impl.hh"

namespace Jetstream::Modules {

struct WebsocketImplNativeCpu : public WebsocketImpl,
                                public Runtime::Context,
                                public Scheduler::Context {
 public:
    Result create() final;

    Result computeSubmit() override;
    Result hasPendingCompute() override;
};

Result WebsocketImplNativeCpu::create() {
    JST_CHECK(WebsocketImpl::create());

    return Result::SUCCESS;
}

Result WebsocketImplNativeCpu::hasPendingCompute() {
    if (circularBuffer.getOccupancy() < buffer.sizeBytes()) {
        return circularBuffer.waitBufferOccupancy(buffer.sizeBytes());
    }

    return Result::SUCCESS;
}

Result WebsocketImplNativeCpu::computeSubmit() {
    if (errored) {
        return Result::ERROR;
    }

    if (circularBuffer.getOccupancy() < buffer.sizeBytes()) {
        return Result::YIELD;
    }

    circularBuffer.get(reinterpret_cast<I8*>(buffer.data()), buffer.sizeBytes());

    return Result::SUCCESS;
}

JST_REGISTER_MODULE(WebsocketImplNativeCpu, DeviceType::CPU, RuntimeType::NATIVE, "generic");

}  // namespace Jetstream::Modules
