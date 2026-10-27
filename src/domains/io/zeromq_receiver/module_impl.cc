#include "module_impl.hh"

#include <limits>
#include <typeinfo>
#include <vector>

#include <zmq.h>

#include <jetstream/memory/attributes.hh>
#include <jetstream/memory/axis.hh>
#include <jetstream/memory/macros.hh>
#include <jetstream/scheduler_context.hh>
#include <jetstream/tools/numeric.hh>

namespace Jetstream::Modules {

namespace {

constexpr long kPollMilliseconds = 200;
constexpr int kHighWaterMark = 8;
constexpr U64 kMaxTrackedStreams = 64;

bool IsVector(const std::any& value) {
    return value.type() == typeid(std::vector<I64>) || value.type() == typeid(std::vector<U64>) ||
           value.type() == typeid(std::vector<F32>) || value.type() == typeid(std::vector<F64>);
}

bool IsAxis(const std::string& key) {
    return key == SampleAxisAttribute || key == BatchAxisAttribute || key == ChannelAxisAttribute;
}

}  // namespace

Result ZeromqReceiverImpl::validate() {
    const auto& config = *candidate();

    validatedDataType = DataType::None;
    validatedCircularBufferSize = 0;

    if (config.pattern != "subscribe" && config.pattern != "pull") {
        JST_ERROR("[MODULE_ZEROMQ_RECEIVER] Invalid pattern '{}'.", config.pattern);
        return Result::ERROR;
    }

    if (config.mode != "connect" && config.mode != "bind") {
        JST_ERROR("[MODULE_ZEROMQ_RECEIVER] Invalid mode '{}'.", config.mode);
        return Result::ERROR;
    }

    if (config.numberOfBatches == 0 || config.numberOfTimeSamples == 0 ||
        config.bufferMultiplier == 0) {
        JST_ERROR("[MODULE_ZEROMQ_RECEIVER] Buffer dimensions must be positive.");
        return Result::ERROR;
    }

    if (config.port == 0 || config.port > std::numeric_limits<U16>::max()) {
        JST_ERROR("[MODULE_ZEROMQ_RECEIVER] Port {} is outside the range 1 to 65535.",
                  config.port);
        return Result::ERROR;
    }

    if (config.dataType != "CF32" && config.dataType != "F32" &&
        config.dataType != "CI8" && config.dataType != "I8" &&
        config.dataType != "CU8" && config.dataType != "U8" &&
        config.dataType != "CI16" && config.dataType != "I16" &&
        config.dataType != "CU16" && config.dataType != "U16") {
        JST_ERROR("[MODULE_ZEROMQ_RECEIVER] Invalid data type '{}'.", config.dataType);
        return Result::ERROR;
    }

    const DataType candidateDataType = NameToDataType(config.dataType);
    U64 outputElements = 0;
    U64 outputSizeBytes = 0;
    U64 circularBufferSize = 0;
    U64 alignedOutputSize = 0;
    if (!detail::CheckedMultiply(config.numberOfBatches,
                                 config.numberOfTimeSamples,
                                 outputElements) ||
        !detail::CheckedMultiply(outputElements,
                                 static_cast<U64>(DataTypeSize(candidateDataType)),
                                 outputSizeBytes) ||
        !detail::CheckedMultiply(outputSizeBytes,
                                 config.bufferMultiplier,
                                 circularBufferSize)) {
        JST_ERROR("[MODULE_ZEROMQ_RECEIVER] Buffer dimensions exceed the supported "
                  "range.");
        return Result::ERROR;
    }

    if (!detail::CheckedPageAlignedSize(outputSizeBytes, alignedOutputSize) ||
        alignedOutputSize > std::numeric_limits<std::size_t>::max() ||
        circularBufferSize > std::numeric_limits<std::size_t>::max()) {
        JST_ERROR("[MODULE_ZEROMQ_RECEIVER] Buffer allocation size is too large.");
        return Result::ERROR;
    }

    validatedDataType = candidateDataType;
    validatedCircularBufferSize = circularBufferSize;
    return Result::SUCCESS;
}

Result ZeromqReceiverImpl::define() {
    JST_CHECK(defineInterfaceOutput("signal"));

    return Result::SUCCESS;
}

Result ZeromqReceiverImpl::create() {
    if (address.empty()) {
        JST_WARN("[MODULE_ZEROMQ_RECEIVER] Address is empty.");
        return Result::INCOMPLETE;
    }

    errored = false;
    lostMessages = 0;
    bufferHealth.publish(0.0f);
    throughputMBs.publish(0.0f);

    JST_CHECK(buffer.create(device(), validatedDataType,
                            {numberOfBatches, numberOfTimeSamples}));
    JST_CHECK(SetSignalAxes(buffer, {
        .sample = Index{1},
        .batch = Index{0},
    }));

    outputs()["signal"].produced(name(), "signal", buffer);

    JST_CHECK(circularBuffer.resize(validatedCircularBufferSize));

    marks.clear();
    pushedBytes = 0;
    appliedSignature.clear();
    appliedKeys.clear();
    streamSequences.clear();
    warnedType = false;
    warnedSize = false;
    announcedHeader = false;
    announcedRaw = false;

    const bool bind = mode == "bind";
    std::string error;
    const ZeroMq::Link::Config config = {
        .type = pattern == "pull" ? ZMQ_PULL : ZMQ_SUB,
        .role = bind ? ZeroMq::Link::Role::Bind : ZeroMq::Link::Role::Connect,
        .address = address,
        .port = static_cast<U16>(port),
        .highWaterMark = kHighWaterMark,
    };
    if (link.open(config, error) != Result::SUCCESS) {
        JST_ERROR("[MODULE_ZEROMQ_RECEIVER] Failed to {} '{}' port {} ({}).",
                  bind ? "listen on" : "connect to", address, port, error);
        return Result::ERROR;
    }

    if (bind) {
        JST_INFO("[MODULE_ZEROMQ_RECEIVER] Listening on '{}' port {}.", address, port);
    } else {
        JST_INFO("[MODULE_ZEROMQ_RECEIVER] Connecting to '{}' port {}.", address, port);
    }

    receiving = true;
    receiveThread.start([this]() { receiveLoop(); });

    return Result::SUCCESS;
}

Result ZeromqReceiverImpl::destroy() {
    receiving = false;
    if (receiveThread.joinable()) {
        receiveThread.join();
    }
    link.close();

    bufferHealth.publish(0.0f);
    throughputMBs.publish(0.0f);

    return Result::SUCCESS;
}

void ZeromqReceiverImpl::receiveLoop() {
    bool warned = false;
    zmq_pollitem_t items[] = {
        {link.socket(), 0, ZMQ_POLLIN, 0},
        {link.monitor(), 0, ZMQ_POLLIN, 0},
    };

    while (receiving) {
        if (zmq_poll(items, 2, kPollMilliseconds) <= 0) {
            continue;
        }

        if (items[1].revents & ZMQ_POLLIN) {
            link.readEvents();
            if (link.takeHandshakeFailure() && !warned) {
                JST_WARN("[MODULE_ZEROMQ_RECEIVER] A peer at '{}' port {} disconnected during "
                         "the handshake. Check that Pattern matches on both ends.", address, port);
                warned = true;
            }
        }

        if (items[0].revents & ZMQ_POLLIN) {
            while (receiving && receiveMessage()) {
            }
        }
    }
}

bool ZeromqReceiverImpl::receiveMessage() {
    void* socket = link.socket();

    zmq_msg_t first;
    zmq_msg_init(&first);
    if (zmq_msg_recv(&first, socket, ZMQ_DONTWAIT) < 0) {
        zmq_msg_close(&first);
        return false;
    }

    if (!zmq_msg_more(&first)) {
        receivePayload(zmq_msg_data(&first), zmq_msg_size(&first), nullptr);
        zmq_msg_close(&first);
        return true;
    }

    zmq_msg_t last;
    zmq_msg_init(&last);
    bool complete = false;
    while (true) {
        zmq_msg_close(&last);
        zmq_msg_init(&last);
        if (zmq_msg_recv(&last, socket, 0) < 0) {
            break;
        }
        if (!zmq_msg_more(&last)) {
            complete = true;
            break;
        }
    }

    if (complete) {
        ZeroMq::Header::Content header;
        const bool decoded = ZeroMq::Header::Decode(zmq_msg_data(&first), zmq_msg_size(&first), header);
        receivePayload(zmq_msg_data(&last), zmq_msg_size(&last), decoded ? &header : nullptr);
    }

    zmq_msg_close(&last);
    zmq_msg_close(&first);
    return complete;
}

void ZeromqReceiverImpl::receivePayload(const void* data,
                                        const U64 size,
                                        const ZeroMq::Header::Content* header) {
    if (size == 0) {
        return;
    }

    if (header) {
        if (header->dtype != dataType) {
            if (!warnedType) {
                JST_WARN("[MODULE_ZEROMQ_RECEIVER] The sender sends {} but Data Type is {}. "
                         "Dropping its buffers.", header->dtype, dataType);
                warnedType = true;
            }
            return;
        }
        if (!announcedHeader) {
            JST_INFO("[MODULE_ZEROMQ_RECEIVER] Receiving {} buffers with attributes.", header->dtype);
            announcedHeader = true;
        }
        if (pattern == "subscribe" && !header->stream.empty()) {
            if (streamSequences.size() >= kMaxTrackedStreams &&
                !streamSequences.contains(header->stream)) {
                streamSequences.clear();
            }
            const auto found = streamSequences.find(header->stream);
            if (found != streamSequences.end() && header->sequence > found->second + 1) {
                lostMessages += header->sequence - found->second - 1;
            }
            streamSequences[header->stream] = header->sequence;
        }
    } else if (!announcedRaw) {
        JST_INFO("[MODULE_ZEROMQ_RECEIVER] Receiving raw buffers as {}.", dataType);
        announcedRaw = true;
    }

    if (size % buffer.elementSize() != 0) {
        if (!warnedSize) {
            JST_WARN("[MODULE_ZEROMQ_RECEIVER] A message of {} bytes does not hold whole {} "
                     "samples. Dropping it.", size, dataType);
            warnedSize = true;
        }
        return;
    }

    {
        std::lock_guard lock(markMutex);
        const std::string signature = header ? header->signature : std::string();
        if (marks.empty() || marks.back().signature != signature) {
            auto attributes = std::make_shared<std::vector<ZeroMq::Header::Attribute>>();
            if (header) {
                *attributes = header->attributes;
            }
            marks.push_back({pushedBytes, signature, std::move(attributes)});
        }

        if (circularBuffer.push(static_cast<const I8*>(data), size) != Result::SUCCESS) {
            errored = true;
            Scheduler::Context::NotifyPendingCompute();
            return;
        }
        pushedBytes += size;

        const U64 oldest = pushedBytes - circularBuffer.size();
        while (marks.size() > 1 && marks[1].start <= oldest) {
            marks.pop_front();
        }
    }
    Scheduler::Context::NotifyPendingCompute();

    const U64 capacity = circularBuffer.capacity();
    if (capacity > 0) {
        const F32 newHealth = static_cast<F32>(circularBuffer.size()) /
                              static_cast<F32>(capacity);
        bufferHealth.publish(bufferHealth.get() * 0.99f + newHealth * 0.01f);
    }

    throughputMBs.publish(static_cast<F32>(circularBuffer.throughput()) / 1e6f);
}

Result ZeromqReceiverImpl::popBuffer() {
    Mark mark;
    {
        std::lock_guard lock(markMutex);
        const auto popped = circularBuffer.pop(reinterpret_cast<I8*>(buffer.data()), buffer.sizeBytes());
        if (popped == Result::INCOMPLETE) {
            return Result::YIELD;
        }
        JST_CHECK(popped);

        const U64 end = pushedBytes - circularBuffer.size();
        while (marks.size() > 1 && marks[1].start < end) {
            marks.pop_front();
        }
        if (!marks.empty()) {
            mark = marks.front();
        }
    }

    if (mark.attributes && mark.signature != appliedSignature) {
        applyAttributes(mark);
    }

    return Result::SUCCESS;
}

void ZeromqReceiverImpl::applyAttributes(const Mark& mark) {
    for (const auto& key : appliedKeys) {
        buffer.removeAttribute(key);
    }
    appliedKeys.clear();

    for (const auto& attribute : *mark.attributes) {
        if (IsAxis(attribute.key) || IsVector(attribute.value)) {
            continue;
        }
        if (ValidateAttribute(attribute.key, attribute.value) != Result::SUCCESS) {
            continue;
        }
        if (buffer.setAttribute(attribute.key, attribute.value) == Result::SUCCESS) {
            appliedKeys.push_back(attribute.key);
        }
    }

    appliedSignature = mark.signature;
}

F32 ZeromqReceiverImpl::getBufferHealth() const {
    return bufferHealth.get();
}

F64 ZeromqReceiverImpl::getBufferLoss() const {
    const auto stats = circularBuffer.statistics();
    return stats.pushedElements > 0
        ? static_cast<F64>(stats.overwrittenElements) / static_cast<F64>(stats.pushedElements)
        : 0.0;
}

F32 ZeromqReceiverImpl::getThroughput() const {
    return throughputMBs.get();
}

U64 ZeromqReceiverImpl::getPeers() const {
    return link.peers();
}

U64 ZeromqReceiverImpl::getLostMessages() const {
    return lostMessages;
}

}  // namespace Jetstream::Modules
