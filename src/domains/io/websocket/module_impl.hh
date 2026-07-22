#ifndef JETSTREAM_DOMAINS_IO_WEBSOCKET_MODULE_IMPL_HH
#define JETSTREAM_DOMAINS_IO_WEBSOCKET_MODULE_IMPL_HH

#include <atomic>
#include <cstddef>
#include <limits>
#include <memory>
#include <mutex>
#include <thread>

#include <jetstream/domains/io/websocket/module.hh>
#include <jetstream/detail/module_impl.hh>
#include <jetstream/tools/circular_buffer.hh>
#include <jetstream/tools/snapshot.hh>

#ifdef JST_OS_BROWSER
#include <emscripten/websocket.h>
#else
#include <httplib.h>
#include <regex>
#endif

namespace Jetstream::Modules {

inline Result ValidateWebsocketConfig(const std::string& url,
                                      const std::string& dataType,
                                      const U64 numberOfBatches,
                                      const U64 numberOfTimeSamples,
                                      const U64 bufferMultiplier) {
    if (numberOfBatches == 0 || numberOfTimeSamples == 0 || bufferMultiplier == 0) {
        JST_ERROR("[MODULE_WEBSOCKET] Buffer dimensions must be positive.");
        return Result::ERROR;
    }

    const DataType type = NameToDataType(dataType);
    if (dataType != "CF32" && dataType != "F32" &&
        dataType != "CI8" && dataType != "I8" &&
        dataType != "CU8" && dataType != "U8" &&
        dataType != "CI16" && dataType != "I16" &&
        dataType != "CU16" && dataType != "U16") {
        JST_ERROR("[MODULE_WEBSOCKET] Invalid data type '{}'.", dataType);
        return Result::ERROR;
    }

    constexpr U64 maxBytes = std::numeric_limits<std::size_t>::max();
    if (numberOfBatches > maxBytes / numberOfTimeSamples) {
        JST_ERROR("[MODULE_WEBSOCKET] Output buffer dimensions are too large.");
        return Result::ERROR;
    }

    const U64 elements = numberOfBatches * numberOfTimeSamples;
    const U64 elementSize = DataTypeSize(type);
    if (elements > maxBytes / elementSize ||
        elements * elementSize > maxBytes / bufferMultiplier) {
        JST_ERROR("[MODULE_WEBSOCKET] Buffer dimensions are too large.");
        return Result::ERROR;
    }

    if (url.empty()) {
        JST_ERROR("[MODULE_WEBSOCKET] URL is empty.");
        return Result::INCOMPLETE;
    }
    if (url.find('#') != std::string::npos) {
        JST_ERROR("[MODULE_WEBSOCKET] WebSocket URL cannot contain a fragment.");
        return Result::ERROR;
    }

#ifdef JST_OS_BROWSER
    const bool ws = url.starts_with("ws://");
    const bool wss = url.starts_with("wss://");
    const std::size_t authorityStart = wss ? 6 : 5;
    if ((!ws && !wss) || authorityStart >= url.size() ||
        url.find_first_of("/?", authorityStart) == authorityStart) {
        JST_ERROR("[MODULE_WEBSOCKET] Invalid WebSocket URL '{}'.", url);
        return Result::ERROR;
    }
#else
    static const std::regex missingPathRegex(R"(^(wss?://[^/?#]+)([?#].*)?$)");
    const std::string clientUrl = std::regex_replace(url, missingPathRegex, "$1/$2");
    try {
        const httplib::ws::WebSocketClient client(clientUrl);
        if (!client.is_valid()) {
            JST_ERROR("[MODULE_WEBSOCKET] Invalid WebSocket URL '{}'.", url);
            return Result::ERROR;
        }
    } catch (const std::exception&) {
        JST_ERROR("[MODULE_WEBSOCKET] Invalid WebSocket URL '{}'.", url);
        return Result::ERROR;
    }
#endif

    return Result::SUCCESS;
}

struct WebsocketImpl : public Module::Impl, public DynamicConfig<Websocket> {
 public:
    Result validate() override;
    Result define() override;
    Result create() override;
    Result destroy() override;

    F32 getBufferHealth() const;
    F32 getThroughput() const;

 protected:
    Tensor buffer;

    std::atomic<bool> connected{false};
    std::atomic<bool> errored{false};

    Tools::CircularBuffer<I8> circularBuffer;
    Tools::Snapshot<F32> bufferHealth{0.0f};
    Tools::Snapshot<F32> throughputMBs{0.0f};

    void receiveBinaryData(const I8* data, U64 numBytes);

#ifdef JST_OS_BROWSER
    EMSCRIPTEN_WEBSOCKET_T websocket = 0;

    static EM_BOOL onOpen(int eventType,
                          const EmscriptenWebSocketOpenEvent* event,
                          void* userData);
    static EM_BOOL onMessage(int eventType,
                             const EmscriptenWebSocketMessageEvent* event,
                             void* userData);
    static EM_BOOL onClose(int eventType,
                           const EmscriptenWebSocketCloseEvent* event,
                           void* userData);
    static EM_BOOL onError(int eventType,
                           const EmscriptenWebSocketErrorEvent* event,
                           void* userData);
#else
    std::unique_ptr<httplib::ws::WebSocketClient> websocket;
    std::thread websocketThread;
    std::mutex websocketMutex;
    std::atomic<bool> websocketRunning{false};

    void websocketLoop();
#endif
};

}  // namespace Jetstream::Modules

#endif  // JETSTREAM_DOMAINS_IO_WEBSOCKET_MODULE_IMPL_HH
