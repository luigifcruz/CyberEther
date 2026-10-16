#pragma once

#include "jetstream/instance_remote.hh"

#include <atomic>
#include <chrono>
#include <cstddef>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include <httplib.h>
#ifdef ERROR
#undef ERROR
#endif
#include <nlohmann/json.hpp>

namespace Jetstream {

namespace Resources {

extern const unsigned char RemoteViewerPageGzip[];
extern const std::size_t RemoteViewerPageGzipSize;

}  // namespace Resources

struct Instance::Remote::Broker {
 public:
    ~Broker();

    Result create(const Config::Broker& config);
    Result destroy();

    bool secure() const;
    const std::string& origin() const;
    const std::string& publicOrigin() const;
    const std::string& producerKey() const;

 private:
    enum class ConnectionKind {
        Producer,
        Listener,
    };

    struct Peer {
        std::string id;
        httplib::ws::WebSocket* socket = nullptr;
        ConnectionKind kind = ConnectionKind::Producer;
        std::string sessionId;
        bool authenticated = false;
        bool closeRequested = false;
        httplib::ws::CloseStatus closeStatus = httplib::ws::CloseStatus::Normal;
        std::string closeReason;
    };

    struct Session {
        std::string id;
        std::string producer;
        std::string consumer;
        bool waiting = true;
    };

    struct Room {
        std::string id;
        std::string producer;
        std::string consumerTokenHash;
        std::vector<Session> sessions;
    };

    httplib::Server::HandlerResponse route(const httplib::Request& request, httplib::Response& response);
    void serve(const httplib::Request& request, httplib::ws::WebSocket& socket);

    void handleMessage(Peer& peer, const nlohmann::json& message);
    void createRoom(Peer& peer);
    void registerListener(Peer& peer, const nlohmann::json& message);
    void approveSession(Peer& peer, const nlohmann::json& message);
    void forwardPeerMessage(Peer& peer, const nlohmann::json& message);
    void handleEndSession(Peer& peer, const nlohmann::json& message);

    void disconnect(const std::string& peerId);
    void endSession(const std::string& sessionId);
    void releaseListener(const std::string& peerId);
    void sendRoomState();
    void send(const std::string& peerId, const nlohmann::json& message);
    void sendError(Peer& peer, const std::string& code, const std::string& details);
    void requestClose(Peer& peer, httplib::ws::CloseStatus status, const std::string& reason);

    bool isProducer(const Peer& peer) const;
    Session* findSession(const std::string& sessionId);

    Config::Broker config;
    std::unique_ptr<httplib::Server> server;
    std::thread thread;
    std::atomic<bool> running = false;
    bool secure_ = false;
    std::string origin_;
    std::string publicOrigin_;
    std::string producerKey_;

    std::mutex mutex;
    std::unordered_map<std::string, Peer> peers;
    std::optional<Room> room;
};

}  // namespace Jetstream
