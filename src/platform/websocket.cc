#include "jetstream/platform.hh"
#include "jetstream/config.hh"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <thread>
#include <vector>

#if defined(JST_OS_BROWSER)
#include <deque>
#include <unordered_map>

#include <emscripten.h>
#include <emscripten/em_asm.h>

EM_JS_DEPS(jst_websocket_deps, "$UTF8ToString,$stringToNewUTF8,malloc,free");
#elif defined(JETSTREAM_LOADER_CPPHTTPLIB_AVAILABLE)
#include <httplib.h>
#ifdef ERROR
#undef ERROR
#endif
#endif

// TODO: Port src/instance_remote_socket.cc and src/instance_remote_impl.hh to Platform::WebSocket.

namespace Jetstream::Platform {

bool WebSocket::Client::NormalizeUrl(const std::string& url, std::string& normalized) {
    normalized.clear();
    if (url.find('#') != std::string::npos) {
        return false;
    }

    const bool secure = url.starts_with("wss://");
    if (!secure && !url.starts_with("ws://")) {
        return false;
    }

    const std::size_t authorityStart = secure ? 6 : 5;
    const std::size_t pathStart = url.find_first_of("/?", authorityStart);
    if (authorityStart >= url.size() || pathStart == authorityStart) {
        return false;
    }

    normalized = url;
    if (pathStart == std::string::npos) {
        normalized += '/';
    } else if (normalized[pathStart] == '?') {
        normalized.insert(pathStart, 1, '/');
    }
    return true;
}

#if defined(JST_OS_BROWSER)

namespace {

struct Channel {
    std::mutex mutex;
    std::condition_variable changed;
    std::deque<std::pair<WebSocket::Message, std::string>> messages;
    bool opened = false;
    bool closed = false;
    bool interrupted = false;
    std::string reason;
};

std::mutex channelsMutex;
std::unordered_map<U32, std::shared_ptr<Channel>> channels;
std::atomic<U32> nextChannel{1};

std::shared_ptr<Channel> FindChannel(U32 id) {
    std::lock_guard lock(channelsMutex);
    const auto found = channels.find(id);
    return found == channels.end() ? nullptr : found->second;
}

void CloseChannel(U32 id) {
    // clang-format off
    MAIN_THREAD_EM_ASM({
        const socket = Module.jstWebSockets?.get($0);
        if (socket && socket.readyState < 2) {
            socket.close(1000);
        }
    }, id);
    // clang-format on
}

}  // namespace

extern "C" {

EMSCRIPTEN_KEEPALIVE
void jst_websocket_on_open(U32 id) {
    if (const auto channel = FindChannel(id)) {
        {
            std::lock_guard lock(channel->mutex);
            channel->opened = true;
        }
        channel->changed.notify_all();
    }
}

EMSCRIPTEN_KEEPALIVE
void jst_websocket_on_message(U32 id, const char* data, U32 length, I32 binary) {
    if (const auto channel = FindChannel(id)) {
        {
            std::lock_guard lock(channel->mutex);
            const auto kind = binary ? WebSocket::Message::Binary : WebSocket::Message::Text;
            channel->messages.emplace_back(kind, std::string(data, length));
        }
        channel->changed.notify_all();
    }
}

EMSCRIPTEN_KEEPALIVE
void jst_websocket_on_close(U32 id, const char* reason) {
    if (const auto channel = FindChannel(id)) {
        {
            std::lock_guard lock(channel->mutex);
            channel->closed = true;
            channel->reason = reason;
        }
        channel->changed.notify_all();
    }
}

}

struct WebSocket::Client::Impl {
    Config config;
    U32 id = 0;
    mutable std::mutex mutex;
    std::shared_ptr<Channel> channel;
    bool interrupted = false;

    std::shared_ptr<Channel> current() const {
        std::lock_guard lock(mutex);
        return channel;
    }

    void release() {
        if (!id) {
            return;
        }
        CloseChannel(id);
        std::lock_guard lock(channelsMutex);
        channels.erase(id);
        id = 0;
    }
};

WebSocket::Client::Client(Config config) : impl(std::make_unique<Impl>()) {
    impl->config = std::move(config);
}

WebSocket::Client::~Client() {
    impl->release();
}

bool WebSocket::Client::valid() const {
    const auto& url = impl->config.url;
    return url.starts_with("ws://") || url.starts_with("wss://");
}

WebSocket::Client::Connection WebSocket::Client::connect() {
    if (!valid()) {
        return {-1, "Invalid WebSocket URL"};
    }
    if (!impl->config.headers.empty()) {
        return {-1, "Browsers cannot send WebSocket headers"};
    }

    impl->release();
    impl->id = nextChannel.fetch_add(1);
    auto created = std::make_shared<Channel>();
    {
        std::lock_guard lock(impl->mutex);
        created->interrupted = impl->interrupted;
        impl->channel = created;
    }
    {
        std::lock_guard lock(channelsMutex);
        channels.emplace(impl->id, created);
    }

    std::string protocols;
    for (const auto& protocol : impl->config.protocols) {
        protocols += (protocols.empty() ? "" : ",") + protocol;
    }

    // clang-format off
    const int started = MAIN_THREAD_EM_ASM_INT({
        const id = $0;
        const protocols = UTF8ToString($2).split(",").filter((protocol) => protocol.length > 0);
        let socket;
        try {
            socket = new WebSocket(UTF8ToString($1), protocols);
        } catch (error) {
            return 0;
        }

        socket.binaryType = 'arraybuffer';
        if (!Module.jstWebSockets) {
            Module.jstWebSockets = new Map();
        }
        Module.jstWebSockets.set(id, socket);

        socket.onopen = () => _jst_websocket_on_open(id);
        socket.onmessage = (event) => {
            const binary = typeof event.data !== 'string';
            const bytes = binary ? new Uint8Array(event.data) : new TextEncoder().encode(event.data);
            const data = _malloc(bytes.length || 1);
            HEAPU8.set(bytes, data);
            _jst_websocket_on_message(id, data, bytes.length, binary ? 1 : 0);
            _free(data);
        };
        socket.onclose = (event) => {
            Module.jstWebSockets.delete(id);
            const reason = stringToNewUTF8(event.reason || 'Connection closed');
            _jst_websocket_on_close(id, reason);
            _free(reason);
        };
        return 1;
    }, impl->id, impl->config.url.c_str(), protocols.c_str());
    // clang-format on

    if (!started) {
        impl->release();
        return {-1, "Invalid WebSocket URL"};
    }

    auto& channel = *created;
    std::unique_lock lock(channel.mutex);
    channel.changed.wait_for(lock, std::chrono::seconds(impl->config.connectTimeoutSeconds), [&] {
        return channel.opened || channel.closed || channel.interrupted;
    });
    if (channel.opened && !channel.closed && !channel.interrupted) {
        return {101, {}};
    }

    std::string reason = "Connection timed out";
    if (channel.interrupted) {
        reason = "Connection interrupted";
    } else if (channel.closed) {
        reason = channel.reason;
    }
    lock.unlock();
    impl->release();
    return {-1, std::move(reason)};
}

WebSocket::Message WebSocket::Client::read(std::string& payload) {
    const auto current = impl->current();
    if (!current) {
        return Message::Closed;
    }

    auto& channel = *current;
    const auto ready = [&] {
        return !channel.messages.empty() || channel.closed || channel.interrupted;
    };

    std::unique_lock lock(channel.mutex);
    if (impl->config.readTimeoutSeconds > 0) {
        if (!channel.changed.wait_for(lock, std::chrono::seconds(impl->config.readTimeoutSeconds), ready)) {
            return Message::Timeout;
        }
    } else {
        channel.changed.wait(lock, ready);
    }

    if (channel.messages.empty()) {
        return Message::Closed;
    }

    auto [kind, data] = std::move(channel.messages.front());
    channel.messages.pop_front();
    payload = std::move(data);
    return kind;
}

bool WebSocket::Client::send(std::string_view data, Message kind) {
    if (!open()) {
        return false;
    }

    // clang-format off
    return MAIN_THREAD_EM_ASM_INT({
        const socket = Module.jstWebSockets?.get($0);
        if (!socket || socket.readyState !== 1) {
            return 0;
        }
        const bytes = HEAPU8.slice($1, $1 + $2);
        socket.send($3 ? bytes : new TextDecoder().decode(bytes));
        return 1;
    }, impl->id, data.data(), data.size(), kind == Message::Binary ? 1 : 0) != 0;
    // clang-format on
}

void WebSocket::Client::close() {
    if (impl->id) {
        CloseChannel(impl->id);
    }
}

void WebSocket::Client::interrupt() {
    std::shared_ptr<Channel> current;
    {
        std::lock_guard lock(impl->mutex);
        impl->interrupted = true;
        current = impl->channel;
    }
    if (!current) {
        return;
    }
    {
        std::lock_guard lock(current->mutex);
        current->interrupted = true;
    }
    current->changed.notify_all();
}

bool WebSocket::Client::open() const {
    const auto current = impl->current();
    if (!current) {
        return false;
    }
    std::lock_guard lock(current->mutex);
    return current->opened && !current->closed;
}

#elif defined(JETSTREAM_LOADER_CPPHTTPLIB_AVAILABLE)

struct WebSocket::Client::Impl {
    std::unique_ptr<httplib::ws::WebSocketClient> client;
    std::atomic<socket_t> handle{INVALID_SOCKET};
    U64 connectTimeoutSeconds = 0;
    U64 readTimeoutSeconds = 0;
};

WebSocket::Client::Client(Config config) : impl(std::make_unique<Impl>()) {
    impl->connectTimeoutSeconds = config.connectTimeoutSeconds;
    impl->readTimeoutSeconds = config.readTimeoutSeconds;

    try {
        httplib::Headers headers(config.headers.begin(), config.headers.end());
        if (!config.protocols.empty()) {
            std::string protocols;
            for (const auto& protocol : config.protocols) {
                protocols += (protocols.empty() ? "" : ", ") + protocol;
            }
            headers.emplace("Sec-WebSocket-Protocol", std::move(protocols));
        }
        impl->client = std::make_unique<httplib::ws::WebSocketClient>(config.url, headers);
    } catch (const std::exception&) {
        impl->client.reset();
        return;
    }
    if (!impl->client->is_valid()) {
        return;
    }

    auto& client = *impl->client;
    client.set_connection_timeout(static_cast<time_t>(config.connectTimeoutSeconds));
    if (config.readTimeoutSeconds > 0) {
        client.set_read_timeout(static_cast<time_t>(config.readTimeoutSeconds));
    }
    client.set_write_timeout(static_cast<time_t>(config.writeTimeoutSeconds));
    client.set_websocket_ping_interval(static_cast<time_t>(config.pingIntervalSeconds));
    if (config.maxMissedPongs > 0) {
        client.set_websocket_max_missed_pongs(static_cast<int>(config.maxMissedPongs));
    }
    client.set_tcp_nodelay(true);
#ifdef CPPHTTPLIB_SSL_ENABLED
    if (!config.verifyCertificate) {
        client.enable_server_certificate_verification(false);
    }
#endif

    client.set_socket_options([impl = impl.get(), bufferBytes = config.bufferBytes](const socket_t socket) {
        if (bufferBytes > 0) {
            const int size = static_cast<int>(bufferBytes);
            httplib::set_socket_opt(socket, SOL_SOCKET, SO_SNDBUF, size);
            httplib::set_socket_opt(socket, SOL_SOCKET, SO_RCVBUF, size);
        }
        DisableSocketSigPipe(static_cast<std::uintptr_t>(socket));
        impl->handle.store(socket);
    });
}

WebSocket::Client::~Client() = default;

bool WebSocket::Client::valid() const {
    return impl->client && impl->client->is_valid();
}

WebSocket::Client::Connection WebSocket::Client::connect() {
    if (!valid()) {
        return {-1, "Invalid WebSocket URL"};
    }

    auto& client = *impl->client;
    const bool unbounded = impl->readTimeoutSeconds == 0;
    if (unbounded) {
        client.set_read_timeout(static_cast<time_t>(impl->connectTimeoutSeconds));
    }
    const auto result = client.connect();
    if (unbounded) {
        client.set_read_timeout(0);
    }
    if (!result) {
        impl->handle.store(INVALID_SOCKET);
        return {result.status(), httplib::to_string(result.error())};
    }
    return {result.status(), {}};
}

WebSocket::Message WebSocket::Client::read(std::string& payload) {
    if (!impl->client) {
        return Message::Closed;
    }

    switch (impl->client->read(payload)) {
        case httplib::ws::Text:
            return Message::Text;
        case httplib::ws::Binary:
            return Message::Binary;
        case httplib::ws::Timeout:
            return Message::Timeout;
        default:
            return Message::Closed;
    }
}

bool WebSocket::Client::send(std::string_view data, Message kind) {
    if (!impl->client) {
        return false;
    }
    if (kind == Message::Binary) {
        return impl->client->send(data.data(), data.size());
    }
    return impl->client->send(std::string(data));
}

void WebSocket::Client::close() {
    if (!impl->client) {
        return;
    }
    impl->client->close();
    (void)impl->client->send("", 0);
}

void WebSocket::Client::interrupt() {
    const socket_t handle = impl->handle.exchange(INVALID_SOCKET);
    if (handle != INVALID_SOCKET) {
        (void)ShutdownSocketRead(static_cast<std::uintptr_t>(handle));
    }
}

bool WebSocket::Client::open() const {
    return impl->client && impl->client->is_open();
}

struct WebSocket::Server::Impl {
    struct Peer {
        httplib::ws::WebSocket* socket = nullptr;
        bool dropped = false;
    };

    static constexpr time_t kReadPollMicroseconds = 200000;

    Config config;
    std::unique_ptr<httplib::Server> server;
    std::thread thread;
    std::atomic<bool> running{false};
    mutable std::mutex mutex;
    std::vector<Peer> peers;

    bool serving(const httplib::ws::WebSocket* socket) const {
        std::lock_guard lock(mutex);
        const auto peer = std::ranges::find(peers, socket, &Peer::socket);
        return running && peer != peers.end() && !peer->dropped;
    }
};

WebSocket::Server::Server(Config config) : impl(std::make_unique<Impl>()) {
    impl->config = std::move(config);
}

WebSocket::Server::~Server() {
    stop();
}

Result WebSocket::Server::start(std::string& error) {
    stop();

    auto server = std::make_unique<httplib::Server>();
    server->set_write_timeout(static_cast<time_t>(impl->config.writeTimeoutSeconds));
    server->set_websocket_ping_interval(static_cast<time_t>(impl->config.pingIntervalSeconds));
    server->set_socket_options([](const socket_t socket) {
#ifdef JST_OS_WINDOWS
        httplib::set_socket_opt(socket, SOL_SOCKET, SO_EXCLUSIVEADDRUSE, 1);
#else
        httplib::set_socket_opt(socket, SOL_SOCKET, SO_REUSEADDR, 1);
#endif
        DisableSocketSigPipe(static_cast<std::uintptr_t>(socket));
    });
    server->WebSocket(R"(/.*)", [impl = impl.get()](const httplib::Request&, httplib::ws::WebSocket& socket) {
        {
            std::lock_guard lock(impl->mutex);
            impl->peers.push_back({.socket = &socket});
        }
        socket.set_read_timeout(0, Impl::kReadPollMicroseconds);
        std::string message;
        while (impl->serving(&socket) && socket.read(message) != httplib::ws::Fail) {
        }
        std::lock_guard lock(impl->mutex);
        std::erase_if(impl->peers, [&](const Impl::Peer& peer) { return peer.socket == &socket; });
    });

    if (!server->bind_to_port(impl->config.address, impl->config.port)) {
        error = "Could not bind the address and port";
        return Result::ERROR;
    }

    impl->server = std::move(server);
    impl->running = true;
    impl->thread = std::thread([server = impl->server.get()]() { server->listen_after_bind(); });
    impl->server->wait_until_ready();
    return Result::SUCCESS;
}

void WebSocket::Server::stop() {
    if (!impl->server) {
        return;
    }
    impl->running = false;
    {
        std::lock_guard lock(impl->mutex);
        for (auto& peer : impl->peers) {
            peer.socket->close(httplib::ws::CloseStatus::GoingAway, "server stopped");
        }
    }
    impl->server->stop();
    if (impl->thread.joinable()) {
        impl->thread.join();
    }
    impl->server.reset();
}

U64 WebSocket::Server::broadcast(std::string_view data, WebSocket::Message kind) {
    std::lock_guard lock(impl->mutex);
    U64 delivered = 0;
    for (auto& peer : impl->peers) {
        if (peer.dropped) {
            continue;
        }
        const bool sent = kind == WebSocket::Message::Binary ? peer.socket->send(data.data(), data.size())
                                                             : peer.socket->send(std::string(data));
        if (sent) {
            delivered += 1;
        } else {
            peer.dropped = true;
        }
    }
    return delivered;
}

U64 WebSocket::Server::clients() const {
    std::lock_guard lock(impl->mutex);
    return std::ranges::count(impl->peers, false, &Impl::Peer::dropped);
}

#else

struct WebSocket::Client::Impl {};

WebSocket::Client::Client(Config) : impl(std::make_unique<Impl>()) {}

WebSocket::Client::~Client() = default;

bool WebSocket::Client::valid() const {
    return false;
}

WebSocket::Client::Connection WebSocket::Client::connect() {
    return {-1, "This build has no WebSocket support"};
}

WebSocket::Message WebSocket::Client::read(std::string&) {
    return Message::Closed;
}

bool WebSocket::Client::send(std::string_view, Message) {
    return false;
}

void WebSocket::Client::close() {}

void WebSocket::Client::interrupt() {}

bool WebSocket::Client::open() const {
    return false;
}

#endif

#if defined(JST_OS_BROWSER) || !defined(JETSTREAM_LOADER_CPPHTTPLIB_AVAILABLE)

struct WebSocket::Server::Impl {};

WebSocket::Server::Server(Config) : impl(std::make_unique<Impl>()) {}

WebSocket::Server::~Server() = default;

Result WebSocket::Server::start(std::string& error) {
    error = "This build cannot host a WebSocket server";
    return Result::ERROR;
}

void WebSocket::Server::stop() {}

U64 WebSocket::Server::broadcast(std::string_view, WebSocket::Message) {
    return 0;
}

U64 WebSocket::Server::clients() const {
    return 0;
}

#endif

}  // namespace Jetstream::Platform
