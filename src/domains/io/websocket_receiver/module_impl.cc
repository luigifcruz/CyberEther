#include "module_impl.hh"

#include <chrono>
#include <limits>
#include <thread>
#include <utility>
#include <vector>

#include <jetstream/memory/axis.hh>
#include <jetstream/memory/macros.hh>
#include <jetstream/scheduler_context.hh>
#include <jetstream/tools/numeric.hh>

namespace Jetstream::Modules {

namespace {

constexpr U64 kSocketBufferBytes = 8 * JST_MB;
constexpr U64 kConnectTimeoutSeconds = 2;
constexpr U64 kReadTimeoutSeconds = 1;
constexpr U64 kWriteTimeoutSeconds = 1;
constexpr auto kReconnectDelay = std::chrono::milliseconds(500);

Platform::WebSocket::Client::Config SocketConfig(const std::string& url) {
    return {
        .url = url,
        .connectTimeoutSeconds = kConnectTimeoutSeconds,
        .readTimeoutSeconds = kReadTimeoutSeconds,
        .writeTimeoutSeconds = kWriteTimeoutSeconds,
        .bufferBytes = kSocketBufferBytes,
        .verifyCertificate = false,
    };
}

}  // namespace

Result WebsocketReceiverImpl::validate() {
    const auto& config = *candidate();

    validatedDataType = DataType::None;
    validatedCircularBufferSize = 0;
    validatedUrl.clear();

    if (config.numberOfBatches == 0 || config.numberOfTimeSamples == 0 ||
        config.bufferMultiplier == 0) {
        JST_ERROR("[MODULE_WEBSOCKET_RECEIVER] Buffer dimensions must be positive.");
        return Result::ERROR;
    }

    if (config.dataType != "CF32" && config.dataType != "F32" &&
        config.dataType != "CI8" && config.dataType != "I8" &&
        config.dataType != "CU8" && config.dataType != "U8" &&
        config.dataType != "CI16" && config.dataType != "I16" &&
        config.dataType != "CU16" && config.dataType != "U16") {
        JST_ERROR("[MODULE_WEBSOCKET_RECEIVER] Invalid data type '{}'.", config.dataType);
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
        JST_ERROR("[MODULE_WEBSOCKET_RECEIVER] Buffer dimensions exceed the supported "
                  "range.");
        return Result::ERROR;
    }

    if (!detail::CheckedPageAlignedSize(outputSizeBytes, alignedOutputSize) ||
        alignedOutputSize > std::numeric_limits<std::size_t>::max() ||
        circularBufferSize > std::numeric_limits<std::size_t>::max()) {
        JST_ERROR("[MODULE_WEBSOCKET_RECEIVER] Buffer allocation size is too large.");
        return Result::ERROR;
    }

    if (config.url.empty()) {
        JST_ERROR("[MODULE_WEBSOCKET_RECEIVER] URL is empty.");
        return Result::INCOMPLETE;
    }

    std::string candidateUrl;
    if (!Platform::WebSocket::Client::NormalizeUrl(config.url, candidateUrl) ||
        !Platform::WebSocket::Client(SocketConfig(candidateUrl)).valid()) {
        JST_ERROR("[MODULE_WEBSOCKET_RECEIVER] Invalid WebSocket URL '{}'.", config.url);
        return Result::ERROR;
    }

    validatedDataType = candidateDataType;
    validatedCircularBufferSize = circularBufferSize;
    validatedUrl = std::move(candidateUrl);
    return Result::SUCCESS;
}

Result WebsocketReceiverImpl::define() {
    JST_CHECK(defineInterfaceOutput("signal"));

    return Result::SUCCESS;
}

Result WebsocketReceiverImpl::create() {
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
    messageMultiplier = bufferMultiplier;

    receiving = true;
    receiveThread.start([this, url = validatedUrl]() { receiveLoop(url); });

    return Result::SUCCESS;
}

Result WebsocketReceiverImpl::destroy() {
    receiving = false;
    {
        std::lock_guard lock(socketMutex);
        if (socket) {
            socket->interrupt();
        }
    }
    if (receiveThread.joinable()) {
        receiveThread.join();
    }

    bufferHealth.publish(0.0f);
    throughputMBs.publish(0.0f);

    return Result::SUCCESS;
}

void WebsocketReceiverImpl::receiveLoop(const std::string& url) {
    bool warned = false;

    while (receiving) {
        auto connection = std::make_shared<Platform::WebSocket::Client>(SocketConfig(url));
        {
            std::lock_guard lock(socketMutex);
            socket = connection;
        }
        if (!receiving) {
            break;
        }

        const auto status = connection->connect();
        if (status) {
            JST_INFO("[MODULE_WEBSOCKET_RECEIVER] Connected to '{}'.", url);
            warned = false;
            circularBuffer.clear();

            std::string payload;
            while (receiving) {
                const auto message = connection->read(payload);
                if (message == Platform::WebSocket::Message::Binary) {
                    receiveBinaryData(reinterpret_cast<const I8*>(payload.data()), payload.size());
                } else if (message == Platform::WebSocket::Message::Closed) {
                    break;
                }
            }

            if (receiving) {
                JST_WARN("[MODULE_WEBSOCKET_RECEIVER] Connection to '{}' closed. Reconnecting.",
                         url);
            }
        } else if (receiving && !warned) {
            JST_WARN("[MODULE_WEBSOCKET_RECEIVER] Failed to connect to '{}' ({}). Retrying.",
                     url, status.error);
            warned = true;
        }

        {
            std::lock_guard lock(socketMutex);
            socket.reset();
        }

        const auto resume = std::chrono::steady_clock::now() + kReconnectDelay;
        while (receiving && std::chrono::steady_clock::now() < resume) {
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
        }
    }
}

void WebsocketReceiverImpl::receiveBinaryData(const I8* data, const U64 numBytes) {
    if (numBytes == 0) {
        return;
    }

    if (numBytes > circularBuffer.capacity() / messageMultiplier) {
        U64 grownSize = 0;
        std::vector<I8> unread(circularBuffer.size());
        while (circularBuffer.pop(unread.data(), unread.size()) == Result::INCOMPLETE) {
            unread.resize(circularBuffer.size());
        }
        if (!detail::CheckedMultiply(numBytes, messageMultiplier, grownSize) ||
            circularBuffer.resize(grownSize) != Result::SUCCESS ||
            circularBuffer.push(unread.data(), unread.size()) != Result::SUCCESS) {
            JST_ERROR("[MODULE_WEBSOCKET_RECEIVER] Failed to grow the buffer for {} byte "
                      "messages.", numBytes);
            errored = true;
            Scheduler::Context::NotifyPendingCompute();
            return;
        }
        JST_INFO("[MODULE_WEBSOCKET_RECEIVER] Grew the buffer to {} bytes to hold {} byte "
                 "messages.", grownSize, numBytes);
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

F32 WebsocketReceiverImpl::getBufferHealth() const {
    return bufferHealth.get();
}

F32 WebsocketReceiverImpl::getThroughput() const {
    return throughputMBs.get();
}

}  // namespace Jetstream::Modules
