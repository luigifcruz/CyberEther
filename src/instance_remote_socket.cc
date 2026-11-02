#include "instance_remote_impl.hh"

#include "jetstream/logger.hh"
#include "jetstream/platform.hh"

#include <chrono>

namespace Jetstream {

namespace {

constexpr std::chrono::seconds kSignallerReadTimeout{5};
constexpr int kSignallerMaxMissedPongs = 2;

}  // namespace

Result Instance::Remote::Impl::createBroker() {
    broker = std::make_unique<Broker>();
    if (broker->create(config.broker) != Result::SUCCESS) {
        broker.reset();
        return Result::ERROR;
    }

    const std::string& brokerOrigin = broker->origin();
    const std::string websocketOrigin = brokerOrigin.starts_with("https://")
                                            ? jst::fmt::format("wss://{}", brokerOrigin.substr(8))
                                            : jst::fmt::format("ws://{}", brokerOrigin.substr(7));
    signallerUrl = jst::fmt::format("{}/api/v1/remote/signaller", websocketOrigin);

    JST_INFO("[REMOTE] Signaller URL: '{}'.", signallerUrl);
    if (startSignaller() != Result::SUCCESS) {
        (void)destroyBroker();
        return Result::ERROR;
    }
    if (createRoom() != Result::SUCCESS) {
        (void)destroyBroker();
        return Result::ERROR;
    }
    inviteUrl_ = jst::fmt::format("{}#{}", clientDomain, consumerToken);

    if (startStream() != Result::SUCCESS) {
        (void)destroyBroker();
        return Result::ERROR;
    }

    return Result::SUCCESS;
}

Result Instance::Remote::Impl::destroyBroker() {
    JST_DEBUG("[REMOTE] Closing broker connection.");
    Result signallerResult = stopSignaller();
    const Result streamResult = stopStream();
    if (broker) {
        if (broker->destroy() != Result::SUCCESS) {
            signallerResult = Result::ERROR;
        }
        broker.reset();
    }
    roomId_.clear();
    consumerToken.clear();
    inviteUrl_.clear();
    clientDomain.clear();
    signallerUrl.clear();
    {
        std::lock_guard<std::mutex> lock(roomMutex);
        signallerReady = false;
        roomReady = false;
        roomFailed = false;
    }
    {
        std::lock_guard<std::mutex> lock(remoteStateMutex);
        waitlist_.clear();
        clients_.clear();
    }
    return signallerResult == Result::SUCCESS && streamResult == Result::SUCCESS
        ? Result::SUCCESS
        : Result::ERROR;
}

Result Instance::Remote::Impl::createRoom() {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    {
        std::unique_lock<std::mutex> lock(roomMutex);
        const bool completed = roomCondition.wait_until(lock, deadline, [this]() {
            return signallerReady || roomFailed || !signallerRunning;
        });
        if (!completed) {
            JST_ERROR("[REMOTE] Timed out while waiting for the signaller welcome message.");
            return Result::ERROR;
        }
        if (!signallerReady) {
            JST_ERROR("[REMOTE] Signaller failed before becoming ready.");
            return Result::ERROR;
        }
    }

    if (!sendSignallerMessage({{"type", "createRoom"}})) {
        JST_ERROR("[REMOTE] Failed to request a remote room.");
        return Result::ERROR;
    }

    std::unique_lock<std::mutex> lock(roomMutex);
    const bool completed = roomCondition.wait_until(lock, deadline, [this]() {
        return roomReady || roomFailed || !signallerRunning;
    });
    if (!completed) {
        JST_ERROR("[REMOTE] Timed out while creating the remote room.");
        return Result::ERROR;
    }
    if (!roomReady) {
        JST_ERROR("[REMOTE] Signaller failed while creating the remote room.");
        return Result::ERROR;
    }

    JST_DEBUG("[REMOTE] New room created.");
    return Result::SUCCESS;
}

Result Instance::Remote::Impl::startSignaller() {
    JST_DEBUG("[REMOTE] Starting WebRTC signaller.");

    {
        std::lock_guard<std::mutex> lock(roomMutex);
        signallerReady = false;
        roomReady = false;
        roomFailed = false;
    }

    if (!connectSignaller(signallerUrl)) {
        JST_ERROR("[REMOTE] Failed to connect to the broker at '{}'.", signallerUrl);
        return Result::ERROR;
    }

    signallerRunning = true;
    try {
        signallerThread = std::thread([this]() { signallerLoop(); });
    } catch (const std::exception& e) {
        JST_ERROR("[REMOTE] Failed to start signaller thread: {}", e.what());
        (void)stopSignaller();
        return Result::ERROR;
    }

    return Result::SUCCESS;
}

Result Instance::Remote::Impl::stopSignaller() {
    signallerRunning = false;
    {
        std::lock_guard<std::mutex> lock(roomMutex);
        if (!roomReady) {
            roomFailed = true;
        }
    }
    roomCondition.notify_all();

    {
        std::lock_guard<std::mutex> lock(signallerMutex);
        if (signallerSocket != INVALID_SOCKET) {
            (void)Platform::ShutdownSocketRead(static_cast<std::uintptr_t>(signallerSocket));
            signallerSocket = INVALID_SOCKET;
        }
    }

    if (signallerThread.joinable()) {
        signallerThread.join();
    }

    closeSignallerClient();

    return Result::SUCCESS;
}

httplib::ws::Result Instance::Remote::Impl::connectSignaller(const std::string& url) {
    auto socket = std::make_shared<socket_t>(INVALID_SOCKET);
    const httplib::Headers headers = {{"Authorization", jst::fmt::format("Bearer {}", broker->producerKey())}};
    auto client = std::make_unique<httplib::ws::WebSocketClient>(url, headers);
    client->set_connection_timeout(5);
    client->set_read_timeout(kSignallerReadTimeout);
    client->set_write_timeout(1);
    client->set_websocket_ping_interval(20);
    client->set_websocket_max_missed_pongs(kSignallerMaxMissedPongs);
    client->set_tcp_nodelay(true);
    if (broker->secure()) {
        client->enable_server_certificate_verification(false);
    }
    client->set_socket_options([socket](const socket_t handle) {
        *socket = handle;
    });

    if (!client->is_valid()) {
        JST_ERROR("[REMOTE] Invalid signaller URL '{}'.", signallerUrl);
        return {};
    }

    auto result = client->connect();
    if (!result) {
        return result;
    }

    std::lock_guard<std::mutex> lock(signallerMutex);
    signallerClient = std::move(client);
    signallerSocket = *socket;
    return result;
}

void Instance::Remote::Impl::closeSignallerClient() {
    std::lock_guard<std::mutex> lock(signallerMutex);
    signallerSocket = INVALID_SOCKET;
    if (signallerClient) {
        signallerClient->close();
        // close() is a no-op when read() marked the WebSocket closed. send()
        // still takes the write lock, draining any in-flight heartbeat.
        (void)signallerClient->send("", 0);
    }
    signallerClient.reset();
}

void Instance::Remote::Impl::signallerLoop() {
    readSignaller();
    closeSignallerClient();
    if (signallerRunning) {
        JST_ERROR("[REMOTE] Broker connection closed.");
    }

    signallerRunning = false;
    {
        std::lock_guard<std::mutex> lock(roomMutex);
        if (!roomReady) {
            roomFailed = true;
        }
    }
    roomCondition.notify_all();
}

void Instance::Remote::Impl::readSignaller() {
    while (signallerRunning) {
        httplib::ws::WebSocketClient* client = nullptr;
        {
            std::lock_guard<std::mutex> lock(signallerMutex);
            if (!signallerClient || !signallerClient->is_open()) {
                return;
            }
            client = signallerClient.get();
        }

        std::string payload;
        const httplib::ws::ReadResult result = client->read(payload);

        if (result == httplib::ws::Text) {
            if (payload.size() > 256 * 1024) {
                JST_ERROR("[REMOTE] Signaller message exceeded the size limit.");
                return;
            }
            handleSignallerMessage(payload);
        } else if (result == httplib::ws::Binary) {
            JST_WARN("[REMOTE] Ignoring binary signaller message.");
        } else if (result != httplib::ws::Timeout) {
            return;
        }
    }
}

bool Instance::Remote::Impl::sendSignallerMessage(const nlohmann::json& j) {
    const std::string payload = j.dump();
    std::lock_guard<std::mutex> lock(signallerMutex);
    if (!signallerClient || !signallerClient->is_open()) {
        return false;
    }
    return signallerClient->send(payload);
}

}  // namespace Jetstream
