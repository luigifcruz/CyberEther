#ifndef JETSTREAM_DOMAINS_IO_ZEROMQ_RECEIVER_MODULE_IMPL_HH
#define JETSTREAM_DOMAINS_IO_ZEROMQ_RECEIVER_MODULE_IMPL_HH

#include <atomic>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include <jetstream/domains/io/zeromq_receiver/module.hh>
#include <jetstream/detail/module_impl.hh>
#include <jetstream/platform.hh>
#include <jetstream/tools/circular_buffer.hh>
#include <jetstream/tools/snapshot.hh>

#include "library/link.hh"

namespace Jetstream::Modules {

struct ZeromqReceiverImpl : public Module::Impl, public DynamicConfig<ZeromqReceiver> {
 public:
    Result validate() override;
    Result define() override;
    Result create() override;
    Result destroy() override;

    F32 getBufferFill() const;
    F64 getBufferLoss() const;
    F32 getThroughput() const;
    U64 getPeers() const;
    U64 getLostMessages() const;

 protected:
    struct Mark {
        U64 start = 0;
        std::string signature;
        std::shared_ptr<const std::vector<ZeroMq::Header::Attribute>> attributes;
    };

    Tensor buffer;

    std::atomic<bool> errored{false};
    std::atomic<bool> receiving{false};

    Tools::CircularBuffer<I8> circularBuffer;
    Tools::Snapshot<F32> bufferFill{0.0f};
    Tools::Snapshot<F32> throughputMBs{0.0f};
    std::atomic<U64> lostMessages{0};

    DataType validatedDataType = DataType::None;
    U64 validatedCircularBufferSize = 0;

    ZeroMq::Link link;
    Platform::WorkerThread receiveThread;

    std::mutex markMutex;
    std::deque<Mark> marks;
    U64 pushedBytes = 0;
    std::string appliedSignature;
    std::vector<std::string> appliedKeys;

    std::unordered_map<std::string, U64> streamSequences;
    bool warnedType = false;
    bool warnedSize = false;
    bool announcedHeader = false;
    bool announcedRaw = false;

    void receiveLoop();
    bool receiveMessage();
    void receivePayload(const void* data, U64 size, const ZeroMq::Header::Content* header);
    Result popBuffer();
    void applyAttributes(const Mark& mark);
};

}  // namespace Jetstream::Modules

#endif  // JETSTREAM_DOMAINS_IO_ZEROMQ_RECEIVER_MODULE_IMPL_HH
