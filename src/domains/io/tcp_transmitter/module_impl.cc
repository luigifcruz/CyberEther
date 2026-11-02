#include "module_impl.hh"

#include <algorithm>
#include <limits>
#include <thread>
#include <utility>

namespace Jetstream::Modules {

namespace {

constexpr U64 kSocketBufferBytes = 8 * JST_MB;
constexpr U64 kConnectTimeoutMilliseconds = 2000;
constexpr U64 kPollMilliseconds = 200;
constexpr U64 kWriteTimeoutMilliseconds = 1000;
constexpr U64 kDrainBytes = 64 * 1024;
constexpr auto kReconnectDelay = std::chrono::milliseconds(500);

}  // namespace

Result TcpTransmitterImpl::validate() {
    const auto& config = *candidate();

    if (config.mode != "server" && config.mode != "client") {
        JST_ERROR("[MODULE_TCP_TRANSMITTER] Invalid mode '{}'.", config.mode);
        return Result::ERROR;
    }

    if (config.port == 0 || config.port > std::numeric_limits<U16>::max()) {
        JST_ERROR("[MODULE_TCP_TRANSMITTER] Port {} is outside the range 1 to 65535.",
                  config.port);
        return Result::ERROR;
    }

    return Result::SUCCESS;
}

Result TcpTransmitterImpl::define() {
    JST_CHECK(defineInterfaceInput("buffer"));

    return Result::SUCCESS;
}

Result TcpTransmitterImpl::create() {
    if (address.empty()) {
        JST_WARN("[MODULE_TCP_TRANSMITTER] Address is empty.");
        return Result::INCOMPLETE;
    }

    throughputMBs.publish(0.0f);
    bytesSinceLastMeasurement = 0;
    lastMeasurementTime = std::chrono::steady_clock::now();

    const Platform::TcpSocket::Config socketConfig = {
        .address = address,
        .port = static_cast<U16>(port),
        .bufferBytes = kSocketBufferBytes,
        .connectTimeoutMilliseconds = kConnectTimeoutMilliseconds,
        .readTimeoutMilliseconds = kPollMilliseconds,
        .writeTimeoutMilliseconds = kWriteTimeoutMilliseconds,
    };

    if (mode == "client") {
        running = true;
        worker.start([this, socketConfig]() { connectLoop(socketConfig); });
        return Result::SUCCESS;
    }

    std::string error;
    if (listener.listen(socketConfig, error) != Result::SUCCESS) {
        JST_ERROR("[MODULE_TCP_TRANSMITTER] Failed to listen on '{}' port {} ({}).",
                  address, port, error);
        return Result::ERROR;
    }

    JST_INFO("[MODULE_TCP_TRANSMITTER] Listening on '{}' port {}.", address, port);

    running = true;
    worker.start([this]() { acceptLoop(); });

    return Result::SUCCESS;
}

Result TcpTransmitterImpl::destroy() {
    running = false;
    if (worker.joinable()) {
        worker.join();
    }
    listener.close();
    {
        std::lock_guard lock(clientsMutex);
        clients.clear();
    }
    {
        std::lock_guard lock(socketMutex);
        peer.reset();
    }
    connected = false;

    throughputMBs.publish(0.0f);
    bytesSinceLastMeasurement = 0;

    return Result::SUCCESS;
}

U64 TcpTransmitterImpl::send(const void* data, const U64 size) {
    if (mode == "server") {
        std::lock_guard lock(clientsMutex);
        U64 delivered = 0;
        for (auto it = clients.begin(); it != clients.end();) {
            if ((*it)->send(data, size) == Result::SUCCESS) {
                delivered += 1;
                ++it;
            } else {
                it = clients.erase(it);
            }
        }
        return delivered;
    }

    std::shared_ptr<Peer> connection;
    {
        std::lock_guard lock(socketMutex);
        connection = peer;
    }
    if (!connection || connection->broken) {
        return 0;
    }

    if (connection->socket.send(data, size) != Result::SUCCESS) {
        connection->broken = true;
        return 0;
    }
    return 1;
}

void TcpTransmitterImpl::acceptLoop() {
    bool warned = false;

    while (running) {
        auto connection = std::make_unique<Platform::TcpSocket>();
        std::string error;
        const Result accepted = listener.accept(*connection, kPollMilliseconds, error);
        {
            std::lock_guard lock(clientsMutex);
            std::erase_if(clients, [](const auto& client) {
                return !client->alive();
            });
        }
        if (accepted == Result::TIMEOUT) {
            continue;
        }
        if (accepted != Result::SUCCESS) {
            if (!warned) {
                JST_WARN("[MODULE_TCP_TRANSMITTER] Failed to accept a client ({}).", error);
                warned = true;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(kPollMilliseconds));
            continue;
        }

        warned = false;
        std::lock_guard lock(clientsMutex);
        clients.push_back(std::move(connection));
    }
}

void TcpTransmitterImpl::connectLoop(const Platform::TcpSocket::Config& config) {
    bool warned = false;
    std::vector<I8> drain(kDrainBytes);

    while (running) {
        auto connection = std::make_shared<Peer>();
        std::string error;
        if (connection->socket.connect(config, error) == Result::SUCCESS) {
            JST_INFO("[MODULE_TCP_TRANSMITTER] Connected to '{}' port {}.",
                     config.address, config.port);
            warned = false;
            {
                std::lock_guard lock(socketMutex);
                peer = connection;
            }
            connected = true;

            while (running && !connection->broken) {
                U64 received = 0;
                const Result result = connection->socket.receive(drain.data(), drain.size(), received);
                if (result != Result::SUCCESS && result != Result::TIMEOUT) {
                    break;
                }
            }

            connection->broken = true;
            connected = false;
            {
                std::lock_guard lock(socketMutex);
                peer.reset();
            }

            if (running) {
                JST_WARN("[MODULE_TCP_TRANSMITTER] Connection to '{}' port {} closed. "
                         "Reconnecting.", config.address, config.port);
            }
        } else if (running && !warned) {
            JST_WARN("[MODULE_TCP_TRANSMITTER] Failed to connect to '{}' port {} ({}). "
                     "Retrying.", config.address, config.port, error);
            warned = true;
        }

        const auto resume = std::chrono::steady_clock::now() + kReconnectDelay;
        while (running && std::chrono::steady_clock::now() < resume) {
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
        }
    }
}

F32 TcpTransmitterImpl::getThroughput() const {
    return throughputMBs.get();
}

U64 TcpTransmitterImpl::getClients() const {
    std::lock_guard lock(clientsMutex);
    return clients.size();
}

bool TcpTransmitterImpl::getConnected() const {
    return connected;
}

void TcpTransmitterImpl::updateThroughput(const U64 deltaBytes) {
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
