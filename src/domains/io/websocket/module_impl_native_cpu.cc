#include <stdexcept>
#include <utility>

#include <jetstream/runtime_context_native_cpu.hh>
#include <jetstream/scheduler_context.hh>
#include <jetstream/module_context.hh>
#include <jetstream/registry.hh>

#include "module_impl.hh"

namespace Jetstream::Modules {

struct WebsocketImplNativeCpu : public WebsocketImpl,
                                public NativeCpuRuntimeContext,
                                public Scheduler::Context {
 public:
    Result validate() final;
    Result create() final;

    Result computeSubmit() override;
    Result hasPendingCompute() override;
};

Result WebsocketImplNativeCpu::validate() {
#ifndef JST_OS_BROWSER
    validatedClient.reset();
#endif

    JST_CHECK(WebsocketImpl::validate());

#ifndef JST_OS_BROWSER
    std::string clientUrl = validatedUrl;
    const std::size_t authorityStart = clientUrl.starts_with("wss://") ? 6 : 5;
    const std::size_t pathStart = clientUrl.find_first_of("/?", authorityStart);
    if (pathStart == std::string::npos) {
        clientUrl += '/';
    } else if (clientUrl[pathStart] == '?') {
        clientUrl.insert(pathStart, 1, '/');
    }

    try {
        auto client = std::make_unique<httplib::ws::WebSocketClient>(clientUrl);
        if (!client->is_valid()) {
            JST_ERROR("[MODULE_WEBSOCKET] Invalid WebSocket URL '{}'.",
                      validatedUrl);
            return Result::ERROR;
        }
        validatedUrl = std::move(clientUrl);
        validatedClient = std::move(client);
    } catch (const std::invalid_argument&) {
        JST_ERROR("[MODULE_WEBSOCKET] Invalid WebSocket URL '{}'.", validatedUrl);
        return Result::ERROR;
    }
#endif

    return Result::SUCCESS;
}

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
