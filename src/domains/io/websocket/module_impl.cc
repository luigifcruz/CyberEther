#include "module_impl.hh"

#include <limits>
#include <utility>

#include <jetstream/memory/axis.hh>
#include <jetstream/memory/macros.hh>
#include <jetstream/tools/numeric.hh>

namespace Jetstream::Modules {

namespace {

Result ValidateWebsocketUrl(const std::string& url,
                            std::string& validatedUrl) {
    validatedUrl.clear();
    if (url.empty()) {
        JST_ERROR("[MODULE_WEBSOCKET] URL is empty.");
        return Result::INCOMPLETE;
    }
    if (url.find('#') != std::string::npos) {
        JST_ERROR("[MODULE_WEBSOCKET] WebSocket URL cannot contain a fragment.");
        return Result::ERROR;
    }

    const bool ws = url.starts_with("ws://");
    const bool wss = url.starts_with("wss://");
    const std::size_t authorityStart = wss ? 6 : 5;
    const std::size_t authorityEnd = url.find_first_of("/?#", authorityStart);
    if ((!ws && !wss) || authorityStart >= url.size() ||
        authorityEnd == authorityStart) {
        JST_ERROR("[MODULE_WEBSOCKET] Invalid WebSocket URL '{}'.", url);
        return Result::ERROR;
    }

    validatedUrl = url;
    return Result::SUCCESS;
}

}  // namespace

Result WebsocketImpl::validate() {
    const auto& config = *candidate();

    validatedDataType = DataType::None;
    validatedCircularBufferSize = 0;
    validatedUrl.clear();

    if (config.numberOfBatches == 0 || config.numberOfTimeSamples == 0 ||
        config.bufferMultiplier == 0) {
        JST_ERROR("[MODULE_WEBSOCKET] Buffer dimensions must be positive.");
        return Result::ERROR;
    }

    if (config.dataType != "CF32" && config.dataType != "F32" &&
        config.dataType != "CI8" && config.dataType != "I8" &&
        config.dataType != "CU8" && config.dataType != "U8" &&
        config.dataType != "CI16" && config.dataType != "I16" &&
        config.dataType != "CU16" && config.dataType != "U16") {
        JST_ERROR("[MODULE_WEBSOCKET] Invalid data type '{}'.", config.dataType);
        return Result::ERROR;
    }

    const DataType candidateDataType = NameToDataType(config.dataType);
    U64 outputElements = 0;
    U64 outputSizeBytes = 0;
    U64 circularBufferSize = 0;
    U64 alignedOutputSize = 0;
    if (!detail::CheckedMultiply(config.numberOfBatches,
                                 config.numberOfTimeSamples,
                                 outputElements) ||
        !detail::CheckedMultiply(outputElements,
                                 static_cast<U64>(DataTypeSize(candidateDataType)),
                                 outputSizeBytes) ||
        !detail::CheckedMultiply(outputSizeBytes,
                                 config.bufferMultiplier,
                                 circularBufferSize)) {
        JST_ERROR("[MODULE_WEBSOCKET] Buffer dimensions exceed the supported "
                  "range.");
        return Result::ERROR;
    }

    if (!detail::CheckedPageAlignedSize(outputSizeBytes, alignedOutputSize) ||
        alignedOutputSize > std::numeric_limits<std::size_t>::max() ||
        circularBufferSize > std::numeric_limits<std::size_t>::max()) {
        JST_ERROR("[MODULE_WEBSOCKET] Buffer allocation size is too large.");
        return Result::ERROR;
    }

    std::string candidateUrl;
    JST_CHECK(ValidateWebsocketUrl(config.url, candidateUrl));

    validatedDataType = candidateDataType;
    validatedCircularBufferSize = circularBufferSize;
    validatedUrl = std::move(candidateUrl);
    return Result::SUCCESS;
}

Result WebsocketImpl::define() {
    JST_CHECK(defineTaint(Module::Taint::BROWSER_MAIN_THREAD));

    JST_CHECK(defineInterfaceOutput("signal"));

    return Result::SUCCESS;
}

Result WebsocketImpl::create() {
    errored = false;
    connected = false;
    bufferHealth.publish(0.0f);
    throughputMBs.publish(0.0f);

    JST_CHECK(buffer.create(device(), validatedDataType,
                            {numberOfBatches, numberOfTimeSamples}));
    JST_CHECK(SetSignalAxes(buffer, {
        .sample = Index{1},
        .batch = Index{0},
    }));

    outputs()["signal"].produced(name(), "signal", buffer);

    JST_CHECK(circularBuffer.resize(validatedCircularBufferSize));

#ifdef JST_OS_BROWSER
    EmscriptenWebSocketCreateAttributes attrs = {
        validatedUrl.c_str(),
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

    JST_INFO("[MODULE_WEBSOCKET] Connecting to '{}'.", validatedUrl);
#else
    auto client = std::move(validatedClient);
    if (!client) {
        JST_ERROR("[MODULE_WEBSOCKET] Missing validated WebSocket client.");
        return Result::ERROR;
    }
    client->set_write_timeout(1);
    client->enable_server_certificate_verification(false);
    client->set_tcp_nodelay(true);

    if (!client->connect()) {
        JST_ERROR("[MODULE_WEBSOCKET] Failed to connect to '{}'.", validatedUrl);
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

    if (circularBuffer.push(data, numBytes) != Result::SUCCESS) {
        errored = true;
        return;
    }

    const U64 capacity = circularBuffer.capacity();
    if (capacity > 0) {
        const F32 newHealth = static_cast<F32>(circularBuffer.size()) /
                              static_cast<F32>(capacity);
        const F32 smoothedHealth = bufferHealth.get() * 0.99f + newHealth * 0.01f;
        bufferHealth.publish(smoothedHealth);
    }

    throughputMBs.publish(static_cast<F32>(circularBuffer.throughput()) / 1e6f);
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
