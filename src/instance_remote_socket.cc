#include "instance_remote_impl.hh"

#include "jetstream/logger.hh"

#include <chrono>

namespace Jetstream {

Result Instance::Remote::Impl::createBroker() {
    JST_INFO("[REMOTE] Connecting to broker at '{}'.", config.broker);

    std::string brokerOrigin = config.broker;
    while (brokerOrigin.size() > 1 && brokerOrigin.back() == '/') {
        brokerOrigin.pop_back();
    }

    std::string websocketOrigin;
    if (brokerOrigin.starts_with("https://")) {
        websocketOrigin = jst::fmt::format("wss://{}", brokerOrigin.substr(8));
    } else if (brokerOrigin.starts_with("http://")) {
        websocketOrigin = jst::fmt::format("ws://{}", brokerOrigin.substr(7));
        JST_WARN("[REMOTE] Broker '{}' uses an unencrypted connection.", config.broker);
    } else {
        JST_ERROR("[REMOTE] Broker URL must use HTTP or HTTPS.");
        return Result::ERROR;
    }

    signallerUrl = jst::fmt::format("{}/api/v1/remote/signaller", websocketOrigin);
    clientDomain = jst::fmt::format("{}/remote", brokerOrigin);

    JST_INFO("[REMOTE] Signaller URL: '{}'.", signallerUrl);
    JST_CHECK(startSignaller());
    if (createRoom() != Result::SUCCESS) {
        (void)stopSignaller();
        return Result::ERROR;
    }
    inviteUrl_ = jst::fmt::format("{}#{}", clientDomain, consumerToken);

    if (startStream() != Result::SUCCESS) {
        (void)stopSignaller();
        return Result::ERROR;
    }

    return Result::SUCCESS;
}

Result Instance::Remote::Impl::destroyBroker() {
    JST_DEBUG("[REMOTE] Closing broker connection.");
    JST_CHECK(stopSignaller());
    JST_CHECK(stopStream());
    roomId_.clear();
    consumerToken.clear();
    inviteUrl_.clear();
    {
        std::lock_guard<std::mutex> lock(remoteStateMutex);
        waitlist_.clear();
        clients_.clear();
    }
    return Result::SUCCESS;
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

    signallerClient = std::make_unique<httplib::ws::WebSocketClient>(signallerUrl);
    signallerClient->set_connection_timeout(5);
    signallerClient->set_write_timeout(1);
    signallerClient->set_websocket_ping_interval(20);
    signallerClient->set_tcp_nodelay(true);

    if (!signallerClient->is_valid() || !signallerClient->connect()) {
        JST_ERROR("[REMOTE] Failed to connect to signaller '{}'.", signallerUrl);
        signallerClient.reset();
        return Result::ERROR;
    }

    signallerRunning = true;
    signallerThread = std::thread([this]() { signallerLoop(); });

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
        if (signallerClient) {
            signallerClient->close();
        }
    }

    if (signallerThread.joinable()) {
        signallerThread.join();
    }

    {
        std::lock_guard<std::mutex> lock(signallerMutex);
        signallerClient.reset();
    }

    return Result::SUCCESS;
}

void Instance::Remote::Impl::signallerLoop() {
    while (signallerRunning) {
        std::string payload;
        httplib::ws::ReadResult result = httplib::ws::Fail;
        httplib::ws::WebSocketClient* client = nullptr;

        {
            std::lock_guard<std::mutex> lock(signallerMutex);
            if (!signallerClient || !signallerClient->is_open()) {
                break;
            }
            client = signallerClient.get();
        }

        if (client) {
            result = client->read(payload);
        }

        if (result == httplib::ws::Text) {
            if (payload.size() > 256 * 1024) {
                JST_ERROR("[REMOTE] Signaller message exceeded the size limit.");
                break;
            }
            handleSignallerMessage(payload);
        } else if (result == httplib::ws::Binary) {
            JST_WARN("[REMOTE] Ignoring binary signaller message.");
        } else if (signallerRunning) {
            JST_ERROR("[REMOTE] Signaller connection closed.");
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

bool Instance::Remote::Impl::sendSignallerMessage(const nlohmann::json& j) {
    const std::string payload = j.dump();
    std::lock_guard<std::mutex> lock(signallerMutex);
    if (!signallerClient || !signallerClient->is_open()) {
        return false;
    }
    return signallerClient->send(payload);
}

}  // namespace Jetstream
