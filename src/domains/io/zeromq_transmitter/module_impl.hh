#ifndef JETSTREAM_DOMAINS_IO_ZEROMQ_TRANSMITTER_MODULE_IMPL_HH
#define JETSTREAM_DOMAINS_IO_ZEROMQ_TRANSMITTER_MODULE_IMPL_HH

#include <atomic>
#include <chrono>
#include <string>

#include <jetstream/domains/io/zeromq_transmitter/module.hh>
#include <jetstream/detail/module_impl.hh>
#include <jetstream/platform.hh>
#include <jetstream/tools/snapshot.hh>

#include "domains/io/zeromq_receiver/library/link.hh"

namespace Jetstream::Modules {

struct ZeromqTransmitterImpl : public Module::Impl, public DynamicConfig<ZeromqTransmitter> {
 public:
    Result validate() override;
    Result define() override;
    Result create() override;
    Result destroy() override;

    F32 getThroughput() const;
    U64 getPeers() const;
    U64 getDroppedBuffers() const;

 protected:
    void send(const Tensor& input);
    void updateThroughput(const U64 deltaBytes);
    void monitorLoop();

    ZeroMq::Link link;
    std::string stream;
    U64 sequence = 0;
    bool warnedDrop = false;
    std::atomic<U64> droppedBuffers{0};

    U64 bytesSinceLastMeasurement = 0;
    std::chrono::steady_clock::time_point lastMeasurementTime{};
    Tools::Snapshot<F32> throughputMBs{0.0f};

    std::atomic<bool> monitoring{false};
    Platform::WorkerThread monitorThread;
};

}  // namespace Jetstream::Modules

#endif  // JETSTREAM_DOMAINS_IO_ZEROMQ_TRANSMITTER_MODULE_IMPL_HH
