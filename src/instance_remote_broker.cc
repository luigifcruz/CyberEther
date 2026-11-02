#include "instance_remote_broker.hh"

#include "jetstream/logger.hh"
#include "jetstream/platform.hh"

#include <openssl/crypto.h>
#include <openssl/evp.h>
#include <openssl/rand.h>
#include <openssl/sha.h>

#include <algorithm>
#include <array>

namespace Jetstream {

namespace {

constexpr const char* kSignallerPath = "/api/v1/remote/signaller";
constexpr const char* kViewerPath = "/remote";
constexpr std::size_t kMaxMessageBytes = 256 * 1024;
constexpr std::size_t kMaxSdpLength = 128 * 1024;
constexpr std::size_t kMaxIceCandidateLength = 8 * 1024;
constexpr std::size_t kMaxTokenLength = 128;
constexpr std::size_t kMaxIdLength = 64;
constexpr std::size_t kMaxSessions = 32;
constexpr std::chrono::seconds kAuthenticationTimeout{10};
constexpr std::chrono::seconds kReadTimeout{1};
constexpr time_t kPingIntervalSeconds = 20;
constexpr int kMaxMissedPongs = 2;

std::string Base64Url(const unsigned char* data, const std::size_t size) {
    std::string encoded(4 * ((size + 2) / 3), '\0');
    const int length = EVP_EncodeBlock(reinterpret_cast<unsigned char*>(encoded.data()), data, static_cast<int>(size));
    encoded.resize(static_cast<std::size_t>(std::max(length, 0)));
    std::replace(encoded.begin(), encoded.end(), '+', '-');
    std::replace(encoded.begin(), encoded.end(), '/', '_');
    while (!encoded.empty() && encoded.back() == '=') {
        encoded.pop_back();
    }
    return encoded;
}

std::string NewToken() {
    std::array<unsigned char, 32> bytes{};
    if (RAND_bytes(bytes.data(), static_cast<int>(bytes.size())) != 1) {
        return {};
    }
    return Base64Url(bytes.data(), bytes.size());
}

std::string NewId() {
    std::array<unsigned char, 16> bytes{};
    if (RAND_bytes(bytes.data(), static_cast<int>(bytes.size())) != 1) {
        return {};
    }
    bytes[6] = static_cast<unsigned char>((bytes[6] & 0x0f) | 0x40);
    bytes[8] = static_cast<unsigned char>((bytes[8] & 0x3f) | 0x80);
    return jst::fmt::format("{:02x}{:02x}{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}-"
                            "{:02x}{:02x}{:02x}{:02x}{:02x}{:02x}",
                            bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
                            bytes[8], bytes[9], bytes[10], bytes[11], bytes[12], bytes[13], bytes[14], bytes[15]);
}

std::string TokenHash(const std::string& token) {
    std::array<unsigned char, SHA256_DIGEST_LENGTH> digest{};
    SHA256(reinterpret_cast<const unsigned char*>(token.data()), token.size(), digest.data());
    return Base64Url(digest.data(), digest.size());
}

bool SameHash(const std::string& a, const std::string& b) {
    return !a.empty() && a.size() == b.size() && CRYPTO_memcmp(a.data(), b.data(), a.size()) == 0;
}

bool IsWildcard(const std::string& address) {
    return address.empty() || address == "0.0.0.0" || address == "::";
}

std::string HostForUrl(const std::string& address) {
    return address.find(':') != std::string::npos ? jst::fmt::format("[{}]", address) : address;
}

bool IsId(const nlohmann::json& value) {
    return value.is_string() && !value.get_ref<const std::string&>().empty() &&
           value.get_ref<const std::string&>().size() <= kMaxIdLength;
}

void Reject(httplib::Response& response, const int status, const std::string& error) {
    response.status = status;
    response.set_content(nlohmann::json{{"error", error}}.dump(), "application/json");
}

}  // namespace

Instance::Remote::Broker::~Broker() {
    (void)destroy();
}

Result Instance::Remote::Broker::create(const Config::Broker& config) {
    if (server) {
        JST_ERROR("[REMOTE] Broker is already running.");
        return Result::ERROR;
    }

    this->config = config;

    if (config.certificate.empty() != config.key.empty()) {
        JST_ERROR("[REMOTE] Broker TLS needs both a certificate and a key.");
        return Result::ERROR;
    }
    if (!config.publicUrl.empty() && !IsRemotePublicUrlValid(config.publicUrl)) {
        JST_ERROR("[REMOTE] Broker public URL must be an HTTP or HTTPS origin without a path.");
        return Result::ERROR;
    }

    producerKey_ = NewToken();
    if (producerKey_.empty()) {
        JST_ERROR("[REMOTE] Failed to create the broker producer key.");
        return Result::ERROR;
    }

    secure_ = !config.certificate.empty();
    if (secure_) {
        server = std::make_unique<httplib::SSLServer>(config.certificate.c_str(), config.key.c_str());
    } else {
        server = std::make_unique<httplib::Server>();
    }
    if (!server->is_valid()) {
        JST_ERROR("[REMOTE] Failed to load the broker TLS certificate or key.");
        server.reset();
        return Result::ERROR;
    }

    server->new_task_queue = [] {
        return new httplib::ThreadPool(8, 2 * kMaxSessions + 16);
    };
    server->set_websocket_ping_interval(kPingIntervalSeconds);
    server->set_websocket_max_missed_pongs(kMaxMissedPongs);
    server->set_pre_routing_handler([this](const httplib::Request& request, httplib::Response& response) {
        return route(request, response);
    });
    server->Get("/", [](const httplib::Request&, httplib::Response& response) {
        response.set_redirect(kViewerPath);
    });
    server->Get(kViewerPath, [](const httplib::Request&, httplib::Response& response) {
        response.set_header("Content-Encoding", "gzip");
        response.set_content(reinterpret_cast<const char*>(Resources::RemoteViewerPageGzip),
                             Resources::RemoteViewerPageGzipSize,
                             "text/html; charset=utf-8");
    });
    server->WebSocket(kSignallerPath, [this](const httplib::Request& request, httplib::ws::WebSocket& socket) {
        serve(request, socket);
    });

    int port = config.port;
    if (port == 0) {
        port = server->bind_to_any_port(config.address);
    } else if (!server->bind_to_port(config.address, port)) {
        port = -1;
    }
    if (port <= 0) {
        JST_ERROR("[REMOTE] Failed to bind the broker to {}:{}.", config.address, config.port);
        server.reset();
        return Result::ERROR;
    }

    const std::string scheme = secure_ ? "https" : "http";
    origin_ = jst::fmt::format("{}://{}:{}",
                               scheme,
                               IsWildcard(config.address) ? "127.0.0.1" : HostForUrl(config.address),
                               port);

    if (!config.publicUrl.empty()) {
        publicOrigin_ = config.publicUrl;
        while (publicOrigin_.size() > 1 && publicOrigin_.back() == '/') {
            publicOrigin_.pop_back();
        }
    } else {
        std::string host = config.address;
        if (IsWildcard(host) && Platform::LocalNetworkAddress(host) != Result::SUCCESS) {
            host = "127.0.0.1";
        }
        publicOrigin_ = jst::fmt::format("{}://{}:{}", scheme, HostForUrl(host), port);
    }

    running = true;
    try {
        thread = std::thread([this]() {
            server->listen_after_bind();
        });
    } catch (const std::exception& e) {
        JST_ERROR("[REMOTE] Failed to start the broker thread: {}", e.what());
        running = false;
        server.reset();
        return Result::ERROR;
    }

    JST_INFO("[REMOTE] Broker listening on {}:{} ({}).", config.address, port, publicOrigin_);
    JST_WARN("[REMOTE] The broker is meant for trusted local networks. Do not expose it directly to the internet.");
    return Result::SUCCESS;
}

Result Instance::Remote::Broker::destroy() {
    if (!server) {
        return Result::SUCCESS;
    }

    running = false;
    {
        std::lock_guard<std::mutex> lock(mutex);
        for (auto& [_, peer] : peers) {
            requestClose(peer, httplib::ws::CloseStatus::GoingAway, "broker stopped");
            peer.socket->close(httplib::ws::CloseStatus::GoingAway, "broker stopped");
        }
    }

    server->stop();
    if (thread.joinable()) {
        thread.join();
    }
    server.reset();

    std::lock_guard<std::mutex> lock(mutex);
    peers.clear();
    room.reset();
    origin_.clear();
    producerKey_.clear();
    publicOrigin_.clear();

    JST_DEBUG("[REMOTE] Broker stopped.");
    return Result::SUCCESS;
}

bool Instance::Remote::Broker::secure() const {
    return secure_;
}

const std::string& Instance::Remote::Broker::origin() const {
    return origin_;
}

const std::string& Instance::Remote::Broker::publicOrigin() const {
    return publicOrigin_;
}

const std::string& Instance::Remote::Broker::producerKey() const {
    return producerKey_;
}

httplib::Server::HandlerResponse Instance::Remote::Broker::route(const httplib::Request& request,
                                                                    httplib::Response& response) {
    if (request.path != kSignallerPath) {
        return httplib::Server::HandlerResponse::Unhandled;
    }

    if (!running) {
        Reject(response, 503, "broker stopped");
        return httplib::Server::HandlerResponse::Handled;
    }

    const auto tokenHashes = request.get_param_value_count("tokenHash");
    if (tokenHashes > 1) {
        Reject(response, 400, "invalid room routing");
        return httplib::Server::HandlerResponse::Handled;
    }

    std::lock_guard<std::mutex> lock(mutex);

    if (tokenHashes == 1) {
        if (!room || !SameHash(request.get_param_value("tokenHash"), room->consumerTokenHash)) {
            Reject(response, 404, "room not found");
            return httplib::Server::HandlerResponse::Handled;
        }
        return httplib::Server::HandlerResponse::Unhandled;
    }

    if (!SameHash(request.get_header_value("Authorization"), "Bearer " + producerKey_)) {
        response.set_header("WWW-Authenticate", "Bearer");
        Reject(response, 401, "producer authorization required");
        return httplib::Server::HandlerResponse::Handled;
    }
    return httplib::Server::HandlerResponse::Unhandled;
}

void Instance::Remote::Broker::serve(const httplib::Request& request, httplib::ws::WebSocket& socket) {
    const std::string peerId = NewId();
    if (peerId.empty()) {
        socket.close(httplib::ws::CloseStatus::InternalError, "failed to create peer");
        return;
    }

    {
        std::lock_guard<std::mutex> lock(mutex);
        if (!running) {
            socket.close(httplib::ws::CloseStatus::GoingAway, "broker stopped");
            return;
        }

        Peer peer;
        peer.id = peerId;
        peer.socket = &socket;
        if (request.has_param("tokenHash")) {
            peer.kind = ConnectionKind::Listener;
        }
        peers.emplace(peerId, std::move(peer));
        send(peerId, {{"type", "welcome"}, {"peerId", peerId}});
    }

    socket.set_read_timeout(kReadTimeout);
    const auto connectedAt = std::chrono::steady_clock::now();
    std::optional<std::pair<httplib::ws::CloseStatus, std::string>> closing;

    while (running && socket.is_open()) {
        std::string payload;
        const httplib::ws::ReadResult result = socket.read(payload);
        const auto now = std::chrono::steady_clock::now();

        std::lock_guard<std::mutex> lock(mutex);
        auto& peer = peers.at(peerId);
        if (peer.closeRequested) {
            closing = std::make_pair(peer.closeStatus, peer.closeReason);
            break;
        }

        if (!peer.authenticated && now - connectedAt >= kAuthenticationTimeout) {
            closing = std::make_pair(httplib::ws::CloseStatus::PolicyViolation, "authentication timeout");
            break;
        }
        if (result == httplib::ws::Timeout) {
            continue;
        }
        if (result == httplib::ws::Binary) {
            closing = std::make_pair(httplib::ws::CloseStatus::UnsupportedData, "binary messages are not supported");
            break;
        }
        if (result != httplib::ws::Text) {
            break;
        }

        if (payload.size() > kMaxMessageBytes) {
            closing = std::make_pair(httplib::ws::CloseStatus::MessageTooBig, "message too large");
            break;
        }

        nlohmann::json message = nlohmann::json::parse(payload, nullptr, false);
        if (message.is_discarded() || !message.is_object() || !message.contains("type") ||
            !message["type"].is_string()) {
            sendError(peer, "invalid_message", "invalid request");
            continue;
        }

        handleMessage(peer, message);
    }

    {
        std::lock_guard<std::mutex> lock(mutex);
        auto it = peers.find(peerId);
        if (it != peers.end() && it->second.closeRequested && !closing) {
            closing = std::make_pair(it->second.closeStatus, it->second.closeReason);
        }
        disconnect(peerId);
    }

    if (socket.is_open()) {
        if (closing) {
            socket.close(closing->first, closing->second);
        } else {
            socket.close();
        }
    }
}

void Instance::Remote::Broker::handleMessage(Peer& peer, const nlohmann::json& message) {
    const auto& type = message["type"].get_ref<const std::string&>();
    if (type == "ping") {
        send(peer.id, {{"type", "pong"}});
    } else if (type == "createRoom") {
        createRoom(peer);
    } else if (type == "setPeerStatus") {
        registerListener(peer, message);
    } else if (type == "approveSession") {
        approveSession(peer, message);
    } else if (type == "peer") {
        forwardPeerMessage(peer, message);
    } else if (type == "endSession") {
        handleEndSession(peer, message);
    } else {
        sendError(peer, "invalid_message", "invalid request");
    }
}

void Instance::Remote::Broker::createRoom(Peer& peer) {
    if (peer.kind != ConnectionKind::Producer) {
        sendError(peer, "invalid_role", "connection is bound to a room");
        return;
    }
    if (peer.authenticated || room) {
        sendError(peer, "already_authenticated", "connection already authenticated");
        return;
    }

    const std::string roomId = NewId();
    const std::string consumerToken = NewToken();
    if (roomId.empty() || consumerToken.empty()) {
        sendError(peer, "room_registration_failed", "failed to register room");
        return;
    }

    room = Room{
        .id = roomId,
        .producer = peer.id,
        .consumerTokenHash = TokenHash(consumerToken),
    };
    peer.authenticated = true;

    send(peer.id, {
        {"type", "roomCreated"},
        {"roomId", roomId},
        {"consumerToken", consumerToken},
        {"clientDomain", publicOrigin_ + kViewerPath},
    });
    sendRoomState();
}

void Instance::Remote::Broker::registerListener(Peer& peer, const nlohmann::json& message) {
    if (peer.kind != ConnectionKind::Listener) {
        sendError(peer, "invalid_role", "invalid role");
        return;
    }
    if (peer.authenticated) {
        sendError(peer, "already_authenticated", "connection already authenticated");
        return;
    }

    const auto& roles = message.contains("roles") ? message["roles"] : nlohmann::json();
    if (!roles.is_array() || roles.size() != 1 || roles[0] != "listener") {
        sendError(peer, "invalid_role", "invalid role");
        return;
    }

    const auto& meta = message.contains("meta") ? message["meta"] : nlohmann::json();
    if (!meta.is_object() || !meta.contains("token") || !meta["token"].is_string() ||
        meta["token"].get_ref<const std::string&>().empty() ||
        meta["token"].get_ref<const std::string&>().size() > kMaxTokenLength) {
        sendError(peer, "invalid_token", "invalid consumer authorization token");
        return;
    }

    if (!room || !SameHash(TokenHash(meta["token"].get<std::string>()), room->consumerTokenHash)) {
        sendError(peer, "invalid_token", "invalid consumer authorization token");
        return;
    }
    if (!peers.contains(room->producer)) {
        sendError(peer, "producer_unavailable", "producer not in the room");
        return;
    }
    if (room->sessions.size() >= kMaxSessions) {
        sendError(peer, "room_full", "room limit reached");
        return;
    }

    const std::string sessionId = NewId();
    if (sessionId.empty()) {
        sendError(peer, "room_full", "failed to create session");
        return;
    }

    room->sessions.push_back({
        .id = sessionId,
        .producer = room->producer,
        .consumer = peer.id,
    });
    peer.authenticated = true;
    peer.sessionId = sessionId;

    send(peer.id, {{"type", "sessionRegistered"}, {"sessionId", sessionId}});
    sendRoomState();
}

void Instance::Remote::Broker::approveSession(Peer& peer, const nlohmann::json& message) {
    if (!isProducer(peer)) {
        sendError(peer, "unauthorized", "unauthorized peer");
        return;
    }

    Session* session = message.contains("sessionId") && IsId(message["sessionId"])
                           ? findSession(message["sessionId"].get<std::string>())
                           : nullptr;
    if (!session || !session->waiting) {
        sendError(peer, "session_not_waiting", "session is not waiting in this room");
        return;
    }

    session->waiting = false;
    const Session started = *session;
    sendRoomState();

    send(started.producer, {
        {"type", "startSession"},
        {"sessionId", started.id},
        {"peerId", started.consumer},
        {"offer", nullptr},
    });
    send(started.consumer, {
        {"type", "sessionStarted"},
        {"sessionId", started.id},
        {"peerId", started.producer},
    });
}

void Instance::Remote::Broker::forwardPeerMessage(Peer& peer, const nlohmann::json& message) {
    if (!message.contains("sessionId") || !IsId(message["sessionId"])) {
        sendError(peer, "invalid_message", "invalid request");
        return;
    }

    const Session* session = findSession(message["sessionId"].get<std::string>());
    if (!session) {
        sendError(peer, "session_not_found", "unknown session");
        return;
    }
    if (session->producer != peer.id && session->consumer != peer.id) {
        sendError(peer, "unauthorized", "unauthorized peer");
        return;
    }
    if (session->waiting) {
        sendError(peer, "session_not_started", "session is waiting");
        return;
    }

    const bool hasSdp = message.contains("sdp") && message["sdp"].is_object();
    const bool hasIce = message.contains("ice") && message["ice"].is_object();
    if (hasSdp == hasIce) {
        sendError(peer, "invalid_message", "invalid request");
        return;
    }

    nlohmann::json forwarded = {{"type", "peer"}, {"sessionId", session->id}};
    if (hasSdp) {
        const auto& sdp = message["sdp"];
        if (!sdp.contains("type") || (sdp["type"] != "offer" && sdp["type"] != "answer") ||
            !sdp.contains("sdp") || !sdp["sdp"].is_string() ||
            sdp["sdp"].get_ref<const std::string&>().empty() ||
            sdp["sdp"].get_ref<const std::string&>().size() > kMaxSdpLength) {
            sendError(peer, "invalid_message", "invalid request");
            return;
        }
        forwarded["sdp"] = {{"type", sdp["type"]}, {"sdp", sdp["sdp"]}};
    } else {
        const auto& ice = message["ice"];
        if (!ice.contains("candidate") || !ice["candidate"].is_string() ||
            ice["candidate"].get_ref<const std::string&>().size() > kMaxIceCandidateLength) {
            sendError(peer, "invalid_message", "invalid request");
            return;
        }
        nlohmann::json index = nullptr;
        if (ice.contains("sdpMLineIndex") && !ice["sdpMLineIndex"].is_null()) {
            if (!ice["sdpMLineIndex"].is_number_unsigned() || ice["sdpMLineIndex"].get<std::uint64_t>() > 65535) {
                sendError(peer, "invalid_message", "invalid request");
                return;
            }
            index = ice["sdpMLineIndex"];
        }
        forwarded["ice"] = {{"candidate", ice["candidate"]}, {"sdpMLineIndex", index}};
    }

    send(session->producer == peer.id ? session->consumer : session->producer, forwarded);
}

void Instance::Remote::Broker::handleEndSession(Peer& peer, const nlohmann::json& message) {
    if (!message.contains("sessionId") || !IsId(message["sessionId"])) {
        sendError(peer, "invalid_message", "invalid request");
        return;
    }

    const Session* session = findSession(message["sessionId"].get<std::string>());
    if (!session) {
        sendError(peer, "session_not_found", "unknown session");
        return;
    }
    if (session->producer != peer.id && session->consumer != peer.id) {
        sendError(peer, "unauthorized", "unauthorized peer");
        return;
    }

    endSession(session->id);
}

void Instance::Remote::Broker::disconnect(const std::string& peerId) {
    const auto it = peers.find(peerId);
    if (it == peers.end()) {
        return;
    }

    if (isProducer(it->second)) {
        const auto sessions = std::move(room->sessions);
        room.reset();
        for (const auto& session : sessions) {
            send(session.consumer, {{"type", "endSession"}, {"sessionId", session.id}});
            releaseListener(session.consumer);
        }
    } else if (!it->second.sessionId.empty()) {
        endSession(it->second.sessionId);
    }

    peers.erase(peerId);
}

void Instance::Remote::Broker::endSession(const std::string& sessionId) {
    if (!room) {
        return;
    }

    const auto it = std::find_if(room->sessions.begin(), room->sessions.end(), [&](const Session& session) {
        return session.id == sessionId;
    });
    if (it == room->sessions.end()) {
        return;
    }

    const Session session = *it;
    room->sessions.erase(it);
    sendRoomState();

    send(session.consumer, {{"type", "endSession"}, {"sessionId", session.id}});
    send(session.producer, {{"type", "endSession"}, {"sessionId", session.id}});
    releaseListener(session.consumer);
}

void Instance::Remote::Broker::releaseListener(const std::string& peerId) {
    const auto it = peers.find(peerId);
    if (it == peers.end() || it->second.kind != ConnectionKind::Listener) {
        return;
    }

    it->second.sessionId.clear();
    requestClose(it->second, httplib::ws::CloseStatus::Normal, "session ended");
}

void Instance::Remote::Broker::sendRoomState() {
    if (!room || room->producer.empty()) {
        return;
    }

    nlohmann::json waiting = nlohmann::json::array();
    nlohmann::json active = nlohmann::json::array();
    for (const auto& session : room->sessions) {
        (session.waiting ? waiting : active).push_back(session.id);
    }

    send(room->producer, {{"type", "roomState"}, {"waiting", waiting}, {"active", active}});
}

void Instance::Remote::Broker::send(const std::string& peerId, const nlohmann::json& message) {
    const auto it = peers.find(peerId);
    if (it == peers.end() || !it->second.socket || it->second.closeRequested) {
        return;
    }

    if (!it->second.socket->send(message.dump())) {
        requestClose(it->second, httplib::ws::CloseStatus::InternalError, "send failed");
    }
}

void Instance::Remote::Broker::sendError(Peer& peer, const std::string& code, const std::string& details) {
    send(peer.id, {{"type", "error"}, {"code", code}, {"details", details}});
}

void Instance::Remote::Broker::requestClose(Peer& peer,
                                               const httplib::ws::CloseStatus status,
                                               const std::string& reason) {
    if (peer.closeRequested) {
        return;
    }

    peer.closeRequested = true;
    peer.closeStatus = status;
    peer.closeReason = reason;
}

bool Instance::Remote::Broker::isProducer(const Peer& peer) const {
    return peer.authenticated && peer.kind != ConnectionKind::Listener && room && room->producer == peer.id;
}

Instance::Remote::Broker::Session* Instance::Remote::Broker::findSession(const std::string& sessionId) {
    if (!room) {
        return nullptr;
    }

    const auto it = std::find_if(room->sessions.begin(), room->sessions.end(), [&](const Session& session) {
        return session.id == sessionId;
    });
    return it == room->sessions.end() ? nullptr : &*it;
}

}  // namespace Jetstream
