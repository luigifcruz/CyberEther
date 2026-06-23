#include "module_impl.hh"

#ifndef JST_OS_BROWSER
#include <regex>
#endif

namespace Jetstream::Modules {

Result WebsocketImpl::validate() {
    const auto& config = *candidate();

    if (config.numberOfBatches == 0) {
        JST_ERROR("[MODULE_WEBSOCKET] Number of batches cannot be zero.");
        return Result::ERROR;
    }

    if (config.numberOfTimeSamples == 0) {
        JST_ERROR("[MODULE_WEBSOCKET] Number of time samples cannot be zero.");
        return Result::ERROR;
    }

    if (config.bufferMultiplier == 0) {
        JST_ERROR("[MODULE_WEBSOCKET] Buffer multiplier cannot be zero.");
        return Result::ERROR;
    }

    if (config.dataType != "CF32" &&
        config.dataType != "F32" &&
        config.dataType != "CI8" &&
        config.dataType != "I8" &&
        config.dataType != "CU8" &&
        config.dataType != "U8" &&
        config.dataType != "CI16" &&
        config.dataType != "I16" &&
        config.dataType != "CU16" &&
        config.dataType != "U16") {
        JST_ERROR("[MODULE_WEBSOCKET] Invalid data type '{}'.", config.dataType);
        return Result::ERROR;
    }

    return Result::SUCCESS;
}

Result WebsocketImpl::define() {
    JST_CHECK(defineTaint(Module::Taint::BROWSER_MAIN_THREAD));

    JST_CHECK(defineInterfaceOutput("signal"));

    return Result::SUCCESS;
}

Result WebsocketImpl::create() {
    if (url.empty()) {
        JST_ERROR("[MODULE_WEBSOCKET] URL is empty.");
        return Result::INCOMPLETE;
    }

    errored = false;
    connected = false;
    bufferHealth.publish(0.0f);
    throughputMBs.publish(0.0f);

    JST_CHECK(buffer.create(device(), NameToDataType(dataType),
                            {numberOfBatches, numberOfTimeSamples}));

    outputs()["signal"].produced(name(), "signal", buffer);

    circularBuffer.resize(buffer.sizeBytes() * bufferMultiplier);

#ifdef JST_OS_BROWSER
    EmscriptenWebSocketCreateAttributes attrs = {
        url.c_str(),
        nullptr,
        EM_TRUE,
    };

    websocket = emscripten_websocket_new(&attrs);
    if (websocket <= 0) {
        JST_ERROR("[MODULE_WEBSOCKET] Failed to create WebSocket.");
        return Result::ERROR;
    }

    emscripten_websocket_set_onopen_callback(websocket, this, onOpen);
    emscripten_websocket_set_onmessage_callback(websocket, this, onMessage);
    emscripten_websocket_set_onclose_callback(websocket, this, onClose);
    emscripten_websocket_set_onerror_callback(websocket, this, onError);

    JST_INFO("[MODULE_WEBSOCKET] Connecting to '{}'.", url);
#else
    // cpp-httplib requires an explicit '/' path; browsers accept ws://host.
    static const std::regex missingPathRegex(R"(^(wss?://[^/?#]+)([?#].*)?$)");
    const std::string clientUrl = std::regex_replace(url, missingPathRegex, "$1/$2");

    auto client = std::make_unique<httplib::ws::WebSocketClient>(clientUrl);
    client->set_write_timeout(1);
    client->enable_server_certificate_verification(false);
    client->set_tcp_nodelay(true);

    if (!client->is_valid()) {
        JST_ERROR("[MODULE_WEBSOCKET] Invalid WebSocket URL '{}'.", url);
        return Result::ERROR;
    }

    if (!client->connect()) {
        JST_ERROR("[MODULE_WEBSOCKET] Failed to connect to '{}'.", url);
        return Result::ERROR;
    }

    connected = true;
    websocketRunning = true;

    {
        std::lock_guard<std::mutex> lock(websocketMutex);
        websocket = std::move(client);
    }

    websocketThread = std::thread([this]() { websocketLoop(); });

    JST_INFO("[MODULE_WEBSOCKET] Connection established.");
#endif

    return Result::SUCCESS;
}

Result WebsocketImpl::destroy() {
#ifdef JST_OS_BROWSER
    if (websocket > 0) {
        emscripten_websocket_close(websocket, 1000, "closing");
        emscripten_websocket_delete(websocket);
        websocket = 0;
    }
#else
    websocketRunning = false;

    {
        std::lock_guard<std::mutex> lock(websocketMutex);
        if (websocket) {
            websocket->close();
        }
    }

    if (websocketThread.joinable()) {
        websocketThread.join();
    }

    {
        std::lock_guard<std::mutex> lock(websocketMutex);
        websocket.reset();
    }
#endif

    connected = false;
    bufferHealth.publish(0.0f);
    throughputMBs.publish(0.0f);

    return Result::SUCCESS;
}

void WebsocketImpl::receiveBinaryData(const I8* data, const U64 numBytes) {
    if (!data || numBytes == 0) {
        return;
    }

    circularBuffer.put(data, numBytes);

    const U64 capacity = circularBuffer.getCapacity();
    if (capacity > 0) {
        const F32 newHealth = static_cast<F32>(circularBuffer.getOccupancy()) /
                              static_cast<F32>(capacity);
        const F32 smoothedHealth = bufferHealth.get() * 0.99f + newHealth * 0.01f;
        bufferHealth.publish(smoothedHealth);
    }

    throughputMBs.publish(static_cast<F32>(circularBuffer.getThroughput()) / 1e6f);
}

#ifdef JST_OS_BROWSER
EM_BOOL WebsocketImpl::onOpen(int,
                              const EmscriptenWebSocketOpenEvent*,
                              void* userData) {
    auto* self = static_cast<WebsocketImpl*>(userData);
    self->connected = true;
    JST_INFO("[MODULE_WEBSOCKET] Connection established.");
    return EM_TRUE;
}

EM_BOOL WebsocketImpl::onMessage(int,
                                 const EmscriptenWebSocketMessageEvent* event,
                                 void* userData) {
    auto* self = static_cast<WebsocketImpl*>(userData);

    if (event->isText || !event->data || event->numBytes == 0) {
        return EM_TRUE;
    }

    self->receiveBinaryData(reinterpret_cast<const I8*>(event->data), event->numBytes);

    return EM_TRUE;
}

EM_BOOL WebsocketImpl::onClose(int,
                               const EmscriptenWebSocketCloseEvent* event,
                               void* userData) {
    auto* self = static_cast<WebsocketImpl*>(userData);
    self->connected = false;
    JST_WARN("[MODULE_WEBSOCKET] Connection closed (code={}).", event->code);
    return EM_TRUE;
}

EM_BOOL WebsocketImpl::onError(int,
                               const EmscriptenWebSocketErrorEvent*,
                               void* userData) {
    auto* self = static_cast<WebsocketImpl*>(userData);
    self->errored = true;
    self->connected = false;
    JST_ERROR("[MODULE_WEBSOCKET] Connection error.");
    return EM_TRUE;
}
#else
void WebsocketImpl::websocketLoop() {
    while (websocketRunning) {
        std::string payload;
        httplib::ws::ReadResult result = httplib::ws::Fail;
        httplib::ws::WebSocketClient* client = nullptr;

        {
            std::lock_guard<std::mutex> lock(websocketMutex);
            if (!websocket || !websocket->is_open()) {
                break;
            }
            client = websocket.get();
        }

        if (client) {
            result = client->read(payload);
        }

        if (result == httplib::ws::Binary) {
            receiveBinaryData(reinterpret_cast<const I8*>(payload.data()), payload.size());
        } else if (result == httplib::ws::Text) {
            continue;
        } else if (websocketRunning) {
            JST_WARN("[MODULE_WEBSOCKET] Connection closed.");
            break;
        }
    }

    connected = false;
}
#endif

F32 WebsocketImpl::getBufferHealth() const {
    return bufferHealth.get();
}

F32 WebsocketImpl::getThroughput() const {
    return throughputMBs.get();
}

}  // namespace Jetstream::Modules
