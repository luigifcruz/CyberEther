#ifndef JETSTREAM_DOMAINS_IO_WEBSOCKET_MODULE_IMPL_HH
#define JETSTREAM_DOMAINS_IO_WEBSOCKET_MODULE_IMPL_HH

#include <atomic>

#include <emscripten/websocket.h>

#include <jetstream/domains/io/websocket/module.hh>
#include <jetstream/detail/module_impl.hh>
#include <jetstream/tools/circular_buffer.hh>

namespace Jetstream::Modules {

struct WebsocketImpl : public Module::Impl, public DynamicConfig<Websocket> {
 public:
    Result validate() override;
    Result define() override;
    Result create() override;
    Result destroy() override;

    Tools::CircularBuffer<I8>& getCircularBuffer();
    const F32& getBufferHealth() const;
    const F32& getThroughput() const;

 protected:
    Tensor buffer;

    EMSCRIPTEN_WEBSOCKET_T websocket = 0;
    std::atomic<bool> connected{false};
    std::atomic<bool> errored{false};

    Tools::CircularBuffer<I8> circularBuffer;
    F32 bufferHealth = 0.0f;
    F32 throughputMBs = 0.0f;

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
};

}  // namespace Jetstream::Modules

#endif  // JETSTREAM_DOMAINS_IO_WEBSOCKET_MODULE_IMPL_HH
