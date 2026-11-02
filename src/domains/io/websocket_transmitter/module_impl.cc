#include "module_impl.hh"

#include <limits>
#include <thread>
#include <utility>

namespace Jetstream::Modules {

namespace {

constexpr U64 kMaxMessageBytes = 16 * JST_MB;
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
        .verifyCertificate = false,
    };
}

}  // namespace

Result WebsocketTransmitterImpl::validate() {
    const auto& config = *candidate();

    validatedUrl.clear();

    if (config.mode == "server") {
        if (config.port == 0 || config.port > std::numeric_limits<U16>::max()) {
            JST_ERROR("[MODULE_WEBSOCKET_TRANSMITTER] Port {} is outside the range 1 to 65535.",
                      config.port);
            return Result::ERROR;
        }
        return Result::SUCCESS;
    }

    if (config.mode != "client") {
        JST_ERROR("[MODULE_WEBSOCKET_TRANSMITTER] Invalid mode '{}'.", config.mode);
        return Result::ERROR;
    }

    if (config.url.empty()) {
        JST_ERROR("[MODULE_WEBSOCKET_TRANSMITTER] URL is empty.");
        return Result::INCOMPLETE;
    }

    std::string candidateUrl;
    if (!Platform::WebSocket::Client::NormalizeUrl(config.url, candidateUrl) ||
        !Platform::WebSocket::Client(SocketConfig(candidateUrl)).valid()) {
        JST_ERROR("[MODULE_WEBSOCKET_TRANSMITTER] Invalid WebSocket URL '{}'.", config.url);
        return Result::ERROR;
    }

    validatedUrl = std::move(candidateUrl);
    return Result::SUCCESS;
}

Result WebsocketTransmitterImpl::define() {
    JST_CHECK(defineInterfaceInput("buffer"));

    return Result::SUCCESS;
}

Result WebsocketTransmitterImpl::create() {
    const U64 messageBytes = inputs().at("buffer").tensor.sizeBytes();
    if (messageBytes > kMaxMessageBytes) {
        JST_ERROR("[MODULE_WEBSOCKET_TRANSMITTER] Input buffer of {} bytes exceeds the "
                  "{} byte message limit.", messageBytes, kMaxMessageBytes);
        return Result::ERROR;
    }

    throughputMBs.publish(0.0f);
    bytesSinceLastMeasurement = 0;
    lastMeasurementTime = std::chrono::steady_clock::now();

    if (mode == "client") {
        connecting = true;
        connectThread.start([this, url = validatedUrl]() { connectLoop(url); });
        return Result::SUCCESS;
    }

    if (address.empty()) {
        JST_WARN("[MODULE_WEBSOCKET_TRANSMITTER] Address is empty.");
        return Result::INCOMPLETE;
    }

    server = std::make_unique<Platform::WebSocket::Server>(Platform::WebSocket::Server::Config{
        .address = address,
        .port = static_cast<U16>(port),
        .writeTimeoutSeconds = kWriteTimeoutSeconds,
    });

    std::string error;
    if (server->start(error) != Result::SUCCESS) {
        JST_ERROR("[MODULE_WEBSOCKET_TRANSMITTER] Failed to listen on '{}' port {} ({}).",
                  address, port, error);
        server.reset();
        return Result::ERROR;
    }

    JST_INFO("[MODULE_WEBSOCKET_TRANSMITTER] Listening on '{}' port {}.", address, port);

    return Result::SUCCESS;
}

Result WebsocketTransmitterImpl::destroy() {
    server.reset();

    connecting = false;
    {
        std::lock_guard lock(socketMutex);
        if (socket) {
            socket->interrupt();
        }
    }
    if (connectThread.joinable()) {
        connectThread.join();
    }
    connected = false;

    throughputMBs.publish(0.0f);
    bytesSinceLastMeasurement = 0;

    return Result::SUCCESS;
}

U64 WebsocketTransmitterImpl::send(const std::string_view data) {
    if (server) {
        return server->broadcast(data, Platform::WebSocket::Message::Binary);
    }

    std::shared_ptr<Platform::WebSocket::Client> connection;
    {
        std::lock_guard lock(socketMutex);
        connection = socket;
    }
    if (!connection || !connected) {
        return 0;
    }

    if (!connection->send(data, Platform::WebSocket::Message::Binary)) {
        connection->interrupt();
        return 0;
    }
    return 1;
}

void WebsocketTransmitterImpl::connectLoop(const std::string& url) {
    bool warned = false;

    while (connecting) {
        auto connection = std::make_shared<Platform::WebSocket::Client>(SocketConfig(url));
        {
            std::lock_guard lock(socketMutex);
            socket = connection;
        }
        if (!connecting) {
            break;
        }

        const auto status = connection->connect();
        if (status) {
            JST_INFO("[MODULE_WEBSOCKET_TRANSMITTER] Connected to '{}'.", url);
            warned = false;
            connected = true;

            std::string payload;
            while (connecting) {
                if (connection->read(payload) == Platform::WebSocket::Message::Closed) {
                    break;
                }
            }

            connected = false;
            if (connecting) {
                JST_WARN("[MODULE_WEBSOCKET_TRANSMITTER] Connection to '{}' closed. Reconnecting.",
                         url);
            }
        } else if (connecting && !warned) {
            JST_WARN("[MODULE_WEBSOCKET_TRANSMITTER] Failed to connect to '{}' ({}). Retrying.",
                     url, status.error);
            warned = true;
        }

        {
            std::lock_guard lock(socketMutex);
            socket.reset();
        }

        const auto resume = std::chrono::steady_clock::now() + kReconnectDelay;
        while (connecting && std::chrono::steady_clock::now() < resume) {
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
        }
    }
}

F32 WebsocketTransmitterImpl::getThroughput() const {
    return throughputMBs.get();
}

U64 WebsocketTransmitterImpl::getClients() const {
    return server ? server->clients() : 0;
}

bool WebsocketTransmitterImpl::getConnected() const {
    return connected;
}

void WebsocketTransmitterImpl::updateThroughput(const U64 deltaBytes) {
    constexpr double kMeasurementPeriodSeconds = 0.10;
    constexpr double kEmaAlpha = 0.3;

    bytesSinceLastMeasurement += deltaBytes;

    const auto now = std::chrono::steady_clock::now();
    const double elapsedSeconds = std::chrono::duration<double>(now - lastMeasurementTime).count();
    if (elapsedSeconds < kMeasurementPeriodSeconds) {
        return;
    }

    const double instantThroughput = static_cast<double>(bytesSinceLastMeasurement) /
                                     1e6 /
                                     elapsedSeconds;
    const double smoothedThroughput = kEmaAlpha * instantThroughput +
                                      (1.0 - kEmaAlpha) * throughputMBs.get();
    throughputMBs.publish(static_cast<F32>(smoothedThroughput));

    bytesSinceLastMeasurement = 0;
    lastMeasurementTime = now;
}

}  // namespace Jetstream::Modules
