#include "module_impl.hh"

#include <limits>
#include <vector>

#include <jetstream/memory/axis.hh>
#include <jetstream/memory/macros.hh>
#include <jetstream/scheduler_context.hh>
#include <jetstream/tools/numeric.hh>

namespace Jetstream::Modules {

namespace {

constexpr U64 kMaxDatagramBytes = 64 * 1024;
constexpr U64 kSocketBufferBytes = 8 * JST_MB;
constexpr U64 kReceiveTimeoutMilliseconds = 100;

}  // namespace

Result UdpReceiverImpl::validate() {
    const auto& config = *candidate();

    validatedDataType = DataType::None;
    validatedCircularBufferSize = 0;

    if (config.numberOfBatches == 0 || config.numberOfTimeSamples == 0 ||
        config.bufferMultiplier == 0) {
        JST_ERROR("[MODULE_UDP_RECEIVER] Buffer dimensions must be positive.");
        return Result::ERROR;
    }

    if (config.port == 0 || config.port > std::numeric_limits<U16>::max()) {
        JST_ERROR("[MODULE_UDP_RECEIVER] Port {} is outside the range 1 to 65535.",
                  config.port);
        return Result::ERROR;
    }

    if (config.dataType != "CF32" && config.dataType != "F32" &&
        config.dataType != "CI8" && config.dataType != "I8" &&
        config.dataType != "CU8" && config.dataType != "U8" &&
        config.dataType != "CI16" && config.dataType != "I16" &&
        config.dataType != "CU16" && config.dataType != "U16") {
        JST_ERROR("[MODULE_UDP_RECEIVER] Invalid data type '{}'.", config.dataType);
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
        JST_ERROR("[MODULE_UDP_RECEIVER] Buffer dimensions exceed the supported "
                  "range.");
        return Result::ERROR;
    }

    if (!detail::CheckedPageAlignedSize(outputSizeBytes, alignedOutputSize) ||
        alignedOutputSize > std::numeric_limits<std::size_t>::max() ||
        circularBufferSize > std::numeric_limits<std::size_t>::max()) {
        JST_ERROR("[MODULE_UDP_RECEIVER] Buffer allocation size is too large.");
        return Result::ERROR;
    }

    validatedDataType = candidateDataType;
    validatedCircularBufferSize = circularBufferSize;
    return Result::SUCCESS;
}

Result UdpReceiverImpl::define() {
    JST_CHECK(defineInterfaceOutput("signal"));

    return Result::SUCCESS;
}

Result UdpReceiverImpl::create() {
    if (filterSource && sourceAddress.empty()) {
        JST_WARN("[MODULE_UDP_RECEIVER] Source address is empty.");
        return Result::INCOMPLETE;
    }

    errored = false;
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

    std::string error;
    const Platform::UdpSocket::Config socketConfig = {
        .address = address,
        .port = static_cast<U16>(port),
        .interfaceName = networkInterface == "auto" ? std::string() : networkInterface,
        .source = filterSource ? sourceAddress : std::string(),
        .bufferBytes = kSocketBufferBytes,
        .timeoutMilliseconds = kReceiveTimeoutMilliseconds,
    };
    if (socket.bind(socketConfig, error) != Result::SUCCESS) {
        JST_ERROR("[MODULE_UDP_RECEIVER] Failed to listen on '{}' port {} ({}).",
                  address, port, error);
        return Result::ERROR;
    }

    receiving = true;
    receiveThread.start([this]() { receiveLoop(); });

    JST_INFO("[MODULE_UDP_RECEIVER] Listening on '{}' port {}.", address, port);

    return Result::SUCCESS;
}

Result UdpReceiverImpl::destroy() {
    receiving = false;
    if (receiveThread.joinable()) {
        receiveThread.join();
    }
    socket.close();

    bufferHealth.publish(0.0f);
    throughputMBs.publish(0.0f);

    return Result::SUCCESS;
}

void UdpReceiverImpl::receiveLoop() {
    std::vector<I8> datagram(kMaxDatagramBytes);

    while (receiving) {
        U64 received = 0;
        const Result result = socket.receive(datagram.data(), datagram.size(), received);
        if (result == Result::TIMEOUT || result == Result::SKIP) {
            continue;
        }
        if (result != Result::SUCCESS) {
            JST_ERROR("[MODULE_UDP_RECEIVER] Failed to receive from '{}' port {}.",
                      address, port);
            errored = true;
            Scheduler::Context::NotifyPendingCompute();
            break;
        }

        receiveBinaryData(datagram.data(), received);
    }
}

void UdpReceiverImpl::receiveBinaryData(const I8* data, const U64 numBytes) {
    if (numBytes == 0) {
        return;
    }

    const auto pushed = circularBuffer.push(data, numBytes);
    Scheduler::Context::NotifyPendingCompute();
    if (pushed != Result::SUCCESS) {
        errored = true;
        return;
    }

    const U64 capacity = circularBuffer.capacity();
    if (capacity > 0) {
        const F32 newHealth = static_cast<F32>(circularBuffer.size()) /
                              static_cast<F32>(capacity);
        const F32 smoothedHealth = bufferHealth.get() * 0.99f + newHealth * 0.01f;
        bufferHealth.publish(smoothedHealth);
    }

    throughputMBs.publish(static_cast<F32>(circularBuffer.throughput()) / 1e6f);
}

F32 UdpReceiverImpl::getBufferHealth() const {
    return bufferHealth.get();
}

F32 UdpReceiverImpl::getThroughput() const {
    return throughputMBs.get();
}

}  // namespace Jetstream::Modules
