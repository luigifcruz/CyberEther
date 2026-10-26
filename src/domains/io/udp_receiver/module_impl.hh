#ifndef JETSTREAM_DOMAINS_IO_UDP_RECEIVER_MODULE_IMPL_HH
#define JETSTREAM_DOMAINS_IO_UDP_RECEIVER_MODULE_IMPL_HH

#include <atomic>

#include <jetstream/domains/io/udp_receiver/module.hh>
#include <jetstream/detail/module_impl.hh>
#include <jetstream/platform.hh>
#include <jetstream/tools/circular_buffer.hh>
#include <jetstream/tools/snapshot.hh>

namespace Jetstream::Modules {

struct UdpReceiverImpl : public Module::Impl, public DynamicConfig<UdpReceiver> {
 public:
    Result validate() override;
    Result define() override;
    Result create() override;
    Result destroy() override;

    F32 getBufferHealth() const;
    F32 getThroughput() const;

 protected:
    Tensor buffer;

    std::atomic<bool> errored{false};
    std::atomic<bool> receiving{false};

    Tools::CircularBuffer<I8> circularBuffer;
    Tools::Snapshot<F32> bufferHealth{0.0f};
    Tools::Snapshot<F32> throughputMBs{0.0f};

    DataType validatedDataType = DataType::None;
    U64 validatedCircularBufferSize = 0;

    Platform::UdpSocket socket;
    Platform::WorkerThread receiveThread;

    void receiveLoop();
    void receiveBinaryData(const I8* data, U64 numBytes);
};

}  // namespace Jetstream::Modules

#endif  // JETSTREAM_DOMAINS_IO_UDP_RECEIVER_MODULE_IMPL_HH
