#ifndef JETSTREAM_DOMAINS_IO_TCP_RECEIVER_MODULE_IMPL_HH
#define JETSTREAM_DOMAINS_IO_TCP_RECEIVER_MODULE_IMPL_HH

#include <atomic>

#include <jetstream/domains/io/tcp_receiver/module.hh>
#include <jetstream/detail/module_impl.hh>
#include <jetstream/platform.hh>
#include <jetstream/tools/circular_buffer.hh>
#include <jetstream/tools/snapshot.hh>

namespace Jetstream::Modules {

struct TcpReceiverImpl : public Module::Impl, public DynamicConfig<TcpReceiver> {
 public:
    Result validate() override;
    Result define() override;
    Result create() override;
    Result destroy() override;

    F32 getBufferFill() const;
    F64 getBufferLoss() const;
    F32 getThroughput() const;

 protected:
    Tensor buffer;

    std::atomic<bool> errored{false};
    std::atomic<bool> receiving{false};

    Tools::CircularBuffer<I8> circularBuffer;
    Tools::Snapshot<F32> bufferFill{0.0f};
    Tools::Snapshot<F32> throughputMBs{0.0f};

    DataType validatedDataType = DataType::None;
    U64 validatedCircularBufferSize = 0;

    Platform::TcpSocket listener;
    Platform::WorkerThread receiveThread;

    void connectLoop(const Platform::TcpSocket::Config& config);
    void acceptLoop();
    void readConnection(Platform::TcpSocket& connection);
    void receiveBinaryData(const I8* data, U64 numBytes);
};

}  // namespace Jetstream::Modules

#endif  // JETSTREAM_DOMAINS_IO_TCP_RECEIVER_MODULE_IMPL_HH
