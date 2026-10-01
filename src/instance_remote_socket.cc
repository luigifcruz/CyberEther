#include "instance_remote_impl.hh"

#include "jetstream/logger.hh"
#include "jetstream/platform.hh"

#include <algorithm>
#include <chrono>
#include <random>

namespace Jetstream {

namespace {

constexpr std::size_t kSignallerConnectAttempts = 3;
constexpr std::chrono::milliseconds kSignallerRetryDelay{250};
constexpr std::chrono::seconds kSignallerReadTimeout{5};
constexpr int kSignallerMaxMissedPongs = 2;
constexpr std::chrono::milliseconds kRejoinInitialDelay{500};
constexpr std::chrono::milliseconds kRejoinMaxDelay{8000};

}  // namespace

Result Instance::Remote::Impl::createBroker() {
    JST_INFO("[REMOTE] Connecting to broker at '{}'.", config.broker);

    if (!IsRemoteBrokerSchemeSupported(config.broker)) {
        JST_ERROR("[REMOTE] Broker URL must use HTTP or HTTPS.");
        return Result::ERROR;
    }

    std::string brokerOrigin = config.broker;
    while (brokerOrigin.size() > 1 && brokerOrigin.back() == '/') {
        brokerOrigin.pop_back();
    }

    std::string websocketOrigin;
    if (brokerOrigin.starts_with("https://")) {
        websocketOrigin = jst::fmt::format("wss://{}", brokerOrigin.substr(8));
    } else {
        websocketOrigin = jst::fmt::format("ws://{}", brokerOrigin.substr(7));
        JST_WARN("[REMOTE] Broker '{}' uses an unencrypted connection.", config.broker);
    }

    signallerUrl = jst::fmt::format("{}/api/v1/remote/signaller", websocketOrigin);
    clientDomain = jst::fmt::format("{}/remote", brokerOrigin);

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
    const Result signallerResult = stopSignaller();
    const Result streamResult = stopStream();
    roomId_.clear();
    consumerToken.clear();
    inviteUrl_.clear();
    clientDomain.clear();
    signallerUrl.clear();
    {
        std::lock_guard<std::mutex> lock(roomMutex);
        producerToken.clear();
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

    for (std::size_t attempt = 1; attempt <= kSignallerConnectAttempts; ++attempt) {
        if (connectSignaller(signallerUrl, {})) {
            break;
        }

        if (attempt == kSignallerConnectAttempts) {
            JST_ERROR("[REMOTE] Failed to connect to signaller '{}' after {} attempts.",
                      signallerUrl,
                      kSignallerConnectAttempts);
            return Result::ERROR;
        }

        const auto retryDelay = kSignallerRetryDelay * (1 << (attempt - 1));
        JST_WARN("[REMOTE] Failed to connect to signaller '{}' (attempt {}/{}). "
                 "Retrying in {} ms.",
                 signallerUrl,
                 attempt,
                 kSignallerConnectAttempts,
                 retryDelay.count());
        std::this_thread::sleep_for(retryDelay);
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

httplib::ws::Result Instance::Remote::Impl::connectSignaller(const std::string& url,
                                                             const httplib::Headers& headers) {
    auto socket = std::make_shared<socket_t>(INVALID_SOCKET);
    auto client = std::make_unique<httplib::ws::WebSocketClient>(url, headers);
    client->set_connection_timeout(5);
    client->set_read_timeout(kSignallerReadTimeout);
    client->set_write_timeout(1);
    client->set_websocket_ping_interval(20);
    client->set_websocket_max_missed_pongs(kSignallerMaxMissedPongs);
    client->set_tcp_nodelay(true);
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
    std::mt19937 rng(std::random_device{}());
    auto deadline = std::chrono::steady_clock::time_point::max();
    auto delay = kRejoinInitialDelay;

    while (signallerRunning) {
        readSignaller(deadline);
        closeSignallerClient();
        if (!signallerRunning) {
            break;
        }

        bool authenticated = false;
        std::string roomId;
        std::string token;
        std::chrono::milliseconds window{};
        {
            std::lock_guard<std::mutex> lock(roomMutex);
            authenticated = roomReady;
            signallerReady = false;
            roomReady = false;
            roomId = roomId_;
            token = producerToken;
            window = rejoinWindow;
        }
        if (token.empty()) {
            JST_ERROR("[REMOTE] Signaller connection closed.");
            break;
        }

        const auto now = std::chrono::steady_clock::now();
        if (authenticated) {
            JST_WARN("[REMOTE] Signaller connection lost. Rejoining room.");
            destroyAllWebRtcSessions();
            {
                std::lock_guard<std::mutex> lock(remoteStateMutex);
                waitlist_.clear();
                clients_.clear();
            }
            deadline = now + window;
            delay = kRejoinInitialDelay;
        } else {
            if (now >= deadline) {
                JST_ERROR("[REMOTE] Reconnect window expired before the remote room was rejoined.");
                break;
            }

            std::uniform_int_distribution<std::chrono::milliseconds::rep> jitter(delay.count() / 2, delay.count());
            const std::chrono::milliseconds retryDelay{jitter(rng)};
            JST_WARN("[REMOTE] Failed to rejoin remote room. Retrying in {} ms.", retryDelay.count());
            {
                std::unique_lock<std::mutex> lock(roomMutex);
                roomCondition.wait_until(lock, std::min(now + retryDelay, deadline), [this]() {
                    return !signallerRunning;
                });
            }
            delay = std::min(delay * 2, kRejoinMaxDelay);
        }

        if (!signallerRunning) {
            break;
        }
        const std::string url = jst::fmt::format("{}?roomId={}", signallerUrl, roomId);
        const httplib::Headers headers = {{"Authorization", jst::fmt::format("Bearer {}", token)}};
        const auto result = connectSignaller(url, headers);
        if (!result && (result.status() == 400 || result.status() == 401 || result.status() == 404)) {
            JST_ERROR("[REMOTE] Broker refused to restore the remote room (HTTP {}).", result.status());
            break;
        }
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

void Instance::Remote::Impl::readSignaller(std::chrono::steady_clock::time_point deadline) {
    std::chrono::steady_clock::time_point lastHeartbeat{};

    while (signallerRunning) {
        httplib::ws::WebSocketClient* client = nullptr;
        {
            std::lock_guard<std::mutex> lock(signallerMutex);
            if (!signallerClient || !signallerClient->is_open()) {
                return;
            }
            client = signallerClient.get();
        }

        const auto now = std::chrono::steady_clock::now();
        if (!roomReady && now >= deadline) {
            return;
        }
        if (heartbeatInterval.count() > 0 && now - lastHeartbeat >= heartbeatInterval) {
            (void)sendSignallerMessage({{"type", "ping"}});
            lastHeartbeat = now;
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
