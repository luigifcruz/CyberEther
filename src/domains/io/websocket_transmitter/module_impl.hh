#ifndef JETSTREAM_DOMAINS_IO_WEBSOCKET_TRANSMITTER_MODULE_IMPL_HH
#define JETSTREAM_DOMAINS_IO_WEBSOCKET_TRANSMITTER_MODULE_IMPL_HH

#include <atomic>
#include <chrono>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>

#include <jetstream/domains/io/websocket_transmitter/module.hh>
#include <jetstream/detail/module_impl.hh>
#include <jetstream/platform.hh>
#include <jetstream/tools/snapshot.hh>

namespace Jetstream::Modules {

struct WebsocketTransmitterImpl : public Module::Impl, public DynamicConfig<WebsocketTransmitter> {
 public:
    Result validate() override;
    Result define() override;
    Result create() override;
    Result destroy() override;

    F32 getThroughput() const;
    U64 getClients() const;
    bool getConnected() const;

 protected:
    U64 send(std::string_view data);
    void updateThroughput(const U64 deltaBytes);

    std::unique_ptr<Platform::WebSocket::Server> server;
    U64 bytesSinceLastMeasurement = 0;
    std::chrono::steady_clock::time_point lastMeasurementTime{};
    Tools::Snapshot<F32> throughputMBs{0.0f};

    std::string validatedUrl;

    std::atomic<bool> connecting{false};
    std::atomic<bool> connected{false};
    std::mutex socketMutex;
    std::shared_ptr<Platform::WebSocket::Client> socket;
    Platform::WorkerThread connectThread;

    void connectLoop(const std::string& url);
};

}  // namespace Jetstream::Modules

#endif  // JETSTREAM_DOMAINS_IO_WEBSOCKET_TRANSMITTER_MODULE_IMPL_HH
