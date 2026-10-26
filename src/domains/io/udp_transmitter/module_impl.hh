#ifndef JETSTREAM_DOMAINS_IO_UDP_TRANSMITTER_MODULE_IMPL_HH
#define JETSTREAM_DOMAINS_IO_UDP_TRANSMITTER_MODULE_IMPL_HH

#include <chrono>

#include <jetstream/domains/io/udp_transmitter/module.hh>
#include <jetstream/detail/module_impl.hh>
#include <jetstream/platform.hh>
#include <jetstream/tools/snapshot.hh>

namespace Jetstream::Modules {

struct UdpTransmitterImpl : public Module::Impl, public DynamicConfig<UdpTransmitter> {
 public:
    Result validate() override;
    Result define() override;
    Result create() override;
    Result destroy() override;

    F32 getThroughput() const;
    U64 getDroppedPackets() const;

 protected:
    void updateThroughput(const U64 deltaBytes);

    Platform::UdpSocket socket;
    U64 payloadBytes = 0;
    bool warnedDrop = false;
    U64 bytesSinceLastMeasurement = 0;
    std::chrono::steady_clock::time_point lastMeasurementTime{};
    Tools::Snapshot<F32> throughputMBs{0.0f};
    Tools::Snapshot<U64> droppedPackets{0};
};

}  // namespace Jetstream::Modules

#endif  // JETSTREAM_DOMAINS_IO_UDP_TRANSMITTER_MODULE_IMPL_HH
