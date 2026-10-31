#ifndef JETSTREAM_DOMAINS_IO_TCP_TRANSMITTER_MODULE_IMPL_HH
#define JETSTREAM_DOMAINS_IO_TCP_TRANSMITTER_MODULE_IMPL_HH

#include <atomic>
#include <chrono>
#include <memory>
#include <mutex>
#include <vector>

#include <jetstream/domains/io/tcp_transmitter/module.hh>
#include <jetstream/detail/module_impl.hh>
#include <jetstream/platform.hh>
#include <jetstream/tools/snapshot.hh>

namespace Jetstream::Modules {

struct TcpTransmitterImpl : public Module::Impl, public DynamicConfig<TcpTransmitter> {
 public:
    Result validate() override;
    Result define() override;
    Result create() override;
    Result destroy() override;

    F32 getThroughput() const;
    U64 getClients() const;
    bool getConnected() const;

 protected:
    U64 send(const void* data, U64 size);
    void updateThroughput(const U64 deltaBytes);

    U64 bytesSinceLastMeasurement = 0;
    std::chrono::steady_clock::time_point lastMeasurementTime{};
    Tools::Snapshot<F32> throughputMBs{0.0f};

    std::atomic<bool> running{false};
    Platform::TcpSocket listener;
    mutable std::mutex clientsMutex;
    std::vector<std::unique_ptr<Platform::TcpSocket>> clients;

    struct Peer {
        Platform::TcpSocket socket;
        std::atomic<bool> broken{false};
    };

    std::atomic<bool> connected{false};
    std::mutex socketMutex;
    std::shared_ptr<Peer> peer;
    Platform::WorkerThread worker;

    void acceptLoop();
    void connectLoop(const Platform::TcpSocket::Config& config);
};

}  // namespace Jetstream::Modules

#endif  // JETSTREAM_DOMAINS_IO_TCP_TRANSMITTER_MODULE_IMPL_HH
