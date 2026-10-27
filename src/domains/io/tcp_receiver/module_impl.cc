#include "module_impl.hh"

#include <algorithm>
#include <chrono>
#include <limits>
#include <thread>
#include <vector>

#include <jetstream/memory/axis.hh>
#include <jetstream/memory/macros.hh>
#include <jetstream/scheduler_context.hh>
#include <jetstream/tools/numeric.hh>

namespace Jetstream::Modules {

namespace {

constexpr U64 kReadChunkBytes = 256 * 1024;
constexpr U64 kSocketBufferBytes = 8 * JST_MB;
constexpr U64 kConnectTimeoutMilliseconds = 2000;
constexpr U64 kPollMilliseconds = 200;
constexpr auto kReconnectDelay = std::chrono::milliseconds(500);

}  // namespace

Result TcpReceiverImpl::validate() {
    const auto& config = *candidate();

    validatedDataType = DataType::None;
    validatedCircularBufferSize = 0;

    if (config.mode != "client" && config.mode != "server") {
        JST_ERROR("[MODULE_TCP_RECEIVER] Invalid mode '{}'.", config.mode);
        return Result::ERROR;
    }

    if (config.numberOfBatches == 0 || config.numberOfTimeSamples == 0 ||
        config.bufferMultiplier == 0) {
        JST_ERROR("[MODULE_TCP_RECEIVER] Buffer dimensions must be positive.");
        return Result::ERROR;
    }

    if (config.port == 0 || config.port > std::numeric_limits<U16>::max()) {
        JST_ERROR("[MODULE_TCP_RECEIVER] Port {} is outside the range 1 to 65535.",
                  config.port);
        return Result::ERROR;
    }

    if (config.dataType != "CF32" && config.dataType != "F32" &&
        config.dataType != "CI8" && config.dataType != "I8" &&
        config.dataType != "CU8" && config.dataType != "U8" &&
        config.dataType != "CI16" && config.dataType != "I16" &&
        config.dataType != "CU16" && config.dataType != "U16") {
        JST_ERROR("[MODULE_TCP_RECEIVER] Invalid data type '{}'.", config.dataType);
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
        JST_ERROR("[MODULE_TCP_RECEIVER] Buffer dimensions exceed the supported "
                  "range.");
        return Result::ERROR;
    }

    if (!detail::CheckedPageAlignedSize(outputSizeBytes, alignedOutputSize) ||
        alignedOutputSize > std::numeric_limits<std::size_t>::max() ||
        circularBufferSize > std::numeric_limits<std::size_t>::max()) {
        JST_ERROR("[MODULE_TCP_RECEIVER] Buffer allocation size is too large.");
        return Result::ERROR;
    }

    validatedDataType = candidateDataType;
    validatedCircularBufferSize = circularBufferSize;
    return Result::SUCCESS;
}

Result TcpReceiverImpl::define() {
    JST_CHECK(defineInterfaceOutput("signal"));

    return Result::SUCCESS;
}

Result TcpReceiverImpl::create() {
    if (address.empty()) {
        JST_WARN("[MODULE_TCP_RECEIVER] Address is empty.");
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

    const Platform::TcpSocket::Config socketConfig = {
        .address = address,
        .port = static_cast<U16>(port),
        .bufferBytes = kSocketBufferBytes,
        .connectTimeoutMilliseconds = kConnectTimeoutMilliseconds,
        .readTimeoutMilliseconds = kPollMilliseconds,
        .writeTimeoutMilliseconds = kPollMilliseconds,
    };

    if (mode == "server") {
        std::string error;
        if (listener.listen(socketConfig, error) != Result::SUCCESS) {
            JST_ERROR("[MODULE_TCP_RECEIVER] Failed to listen on '{}' port {} ({}).",
                      address, port, error);
            return Result::ERROR;
        }

        JST_INFO("[MODULE_TCP_RECEIVER] Listening on '{}' port {}.", address, port);

        receiving = true;
        receiveThread.start([this]() { acceptLoop(); });
        return Result::SUCCESS;
    }

    receiving = true;
    receiveThread.start([this, socketConfig]() { connectLoop(socketConfig); });

    return Result::SUCCESS;
}

Result TcpReceiverImpl::destroy() {
    receiving = false;
    if (receiveThread.joinable()) {
        receiveThread.join();
    }
    listener.close();

    bufferHealth.publish(0.0f);
    throughputMBs.publish(0.0f);

    return Result::SUCCESS;
}

void TcpReceiverImpl::connectLoop(const Platform::TcpSocket::Config& config) {
    bool warned = false;

    while (receiving) {
        Platform::TcpSocket connection;
        std::string error;
        if (connection.connect(config, error) == Result::SUCCESS) {
            JST_INFO("[MODULE_TCP_RECEIVER] Connected to '{}' port {}.",
                     config.address, config.port);
            warned = false;
            readConnection(connection);

            if (receiving) {
                JST_WARN("[MODULE_TCP_RECEIVER] Connection to '{}' port {} closed. "
                         "Reconnecting.", config.address, config.port);
            }
        } else if (receiving && !warned) {
            JST_WARN("[MODULE_TCP_RECEIVER] Failed to connect to '{}' port {} ({}). "
                     "Retrying.", config.address, config.port, error);
            warned = true;
        }

        const auto resume = std::chrono::steady_clock::now() + kReconnectDelay;
        while (receiving && std::chrono::steady_clock::now() < resume) {
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
        }
    }
}

void TcpReceiverImpl::acceptLoop() {
    bool warned = false;

    while (receiving) {
        Platform::TcpSocket connection;
        std::string error;
        const Result accepted = listener.accept(connection, kPollMilliseconds, error);
        if (accepted == Result::TIMEOUT) {
            continue;
        }
        if (accepted != Result::SUCCESS) {
            if (!warned) {
                JST_WARN("[MODULE_TCP_RECEIVER] Failed to accept a sender ({}).", error);
                warned = true;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(kPollMilliseconds));
            continue;
        }

        JST_INFO("[MODULE_TCP_RECEIVER] Sender connected on port {}.", port);
        warned = false;
        readConnection(connection);

        if (receiving) {
            JST_WARN("[MODULE_TCP_RECEIVER] Connection on port {} closed. Waiting for the "
                     "next sender.", port);
        }
    }
}

void TcpReceiverImpl::readConnection(Platform::TcpSocket& connection) {
    const U64 sampleBytes = buffer.elementSize();
    std::vector<I8> chunk(kReadChunkBytes + sampleBytes);
    U64 carried = 0;
    circularBuffer.clear();

    while (receiving) {
        U64 received = 0;
        const Result result = connection.receive(chunk.data() + carried, kReadChunkBytes, received);
        if (result == Result::TIMEOUT) {
            continue;
        }
        if (result != Result::SUCCESS) {
            return;
        }

        const U64 total = carried + received;
        const U64 whole = total - total % sampleBytes;
        receiveBinaryData(chunk.data(), whole);
        carried = total - whole;
        std::copy(chunk.begin() + static_cast<std::ptrdiff_t>(whole),
                  chunk.begin() + static_cast<std::ptrdiff_t>(total),
                  chunk.begin());
    }
}

void TcpReceiverImpl::receiveBinaryData(const I8* data, const U64 numBytes) {
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

F32 TcpReceiverImpl::getBufferHealth() const {
    return bufferHealth.get();
}

F64 TcpReceiverImpl::getBufferLoss() const {
    const auto stats = circularBuffer.statistics();
    return stats.pushedElements > 0
        ? static_cast<F64>(stats.overwrittenElements) / static_cast<F64>(stats.pushedElements)
        : 0.0;
}

F32 TcpReceiverImpl::getThroughput() const {
    return throughputMBs.get();
}

}  // namespace Jetstream::Modules
