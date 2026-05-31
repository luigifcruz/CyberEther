#ifndef JETSTREAM_DOMAINS_IO_WEBSOCKET_MODULE_IMPL_HH
#define JETSTREAM_DOMAINS_IO_WEBSOCKET_MODULE_IMPL_HH

#include <atomic>
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
#endif

namespace Jetstream::Modules {

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
