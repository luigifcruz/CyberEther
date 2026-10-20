#include "jetstream/platform.hh"
#include "jetstream/config.hh"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <mutex>

#if defined(JST_OS_BROWSER)
#include <deque>
#include <unordered_map>

#include <emscripten.h>
#include <emscripten/em_asm.h>

EM_JS_DEPS(jst_websocket_deps, "$UTF8ToString,$stringToNewUTF8,malloc,free");
#elif defined(JETSTREAM_LOADER_CPPHTTPLIB_AVAILABLE)
#include <httplib.h>
#endif

// TODO: Port src/instance_remote_socket.cc and src/instance_remote_impl.hh to Platform::WebSocket.
// TODO: Port src/domains/io/websocket/module_impl.cc, module_impl.hh, and module_impl_native_cpu.cc to Platform::WebSocket.

namespace Jetstream::Platform {

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

struct WebSocket::Impl {
    Config config;
    U32 id = 0;
    std::shared_ptr<Channel> channel;

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

WebSocket::WebSocket(Config config) : impl(std::make_unique<Impl>()) {
    impl->config = std::move(config);
}

WebSocket::~WebSocket() {
    impl->release();
}

bool WebSocket::valid() const {
    const auto& url = impl->config.url;
    return url.starts_with("ws://") || url.starts_with("wss://");
}

WebSocket::Connection WebSocket::connect() {
    if (!valid()) {
        return {-1, "Invalid WebSocket URL"};
    }
    if (!impl->config.headers.empty()) {
        return {-1, "Browsers cannot send WebSocket headers"};
    }

    impl->release();
    impl->id = nextChannel.fetch_add(1);
    impl->channel = std::make_shared<Channel>();
    {
        std::lock_guard lock(channelsMutex);
        channels.emplace(impl->id, impl->channel);
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

    auto& channel = *impl->channel;
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

WebSocket::Message WebSocket::read(std::string& payload) {
    if (!impl->channel) {
        return Message::Closed;
    }

    auto& channel = *impl->channel;
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

bool WebSocket::send(std::string_view text) {
    if (!open()) {
        return false;
    }

    // clang-format off
    return MAIN_THREAD_EM_ASM_INT({
        const socket = Module.jstWebSockets?.get($0);
        if (!socket || socket.readyState !== 1) {
            return 0;
        }
        socket.send(new TextDecoder().decode(HEAPU8.slice($1, $1 + $2)));
        return 1;
    }, impl->id, text.data(), text.size()) != 0;
    // clang-format on
}

void WebSocket::close() {
    if (impl->id) {
        CloseChannel(impl->id);
    }
}

void WebSocket::interrupt() {
    if (!impl->channel) {
        return;
    }
    {
        std::lock_guard lock(impl->channel->mutex);
        impl->channel->interrupted = true;
    }
    impl->channel->changed.notify_all();
}

bool WebSocket::open() const {
    if (!impl->channel) {
        return false;
    }
    std::lock_guard lock(impl->channel->mutex);
    return impl->channel->opened && !impl->channel->closed;
}

#elif defined(JETSTREAM_LOADER_CPPHTTPLIB_AVAILABLE)

struct WebSocket::Impl {
    std::unique_ptr<httplib::ws::WebSocketClient> client;
    std::atomic<socket_t> handle{INVALID_SOCKET};
    U64 connectTimeoutSeconds = 0;
    U64 readTimeoutSeconds = 0;
};

WebSocket::WebSocket(Config config) : impl(std::make_unique<Impl>()) {
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

WebSocket::~WebSocket() = default;

bool WebSocket::valid() const {
    return impl->client && impl->client->is_valid();
}

WebSocket::Connection WebSocket::connect() {
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

WebSocket::Message WebSocket::read(std::string& payload) {
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

bool WebSocket::send(std::string_view text) {
    return impl->client && impl->client->send(std::string(text));
}

void WebSocket::close() {
    if (!impl->client) {
        return;
    }
    impl->client->close();
    (void)impl->client->send("", 0);
}

void WebSocket::interrupt() {
    const socket_t handle = impl->handle.exchange(INVALID_SOCKET);
    if (handle != INVALID_SOCKET) {
        (void)ShutdownSocketRead(static_cast<std::uintptr_t>(handle));
    }
}

bool WebSocket::open() const {
    return impl->client && impl->client->is_open();
}

#else

struct WebSocket::Impl {};

WebSocket::WebSocket(Config) : impl(std::make_unique<Impl>()) {}

WebSocket::~WebSocket() = default;

bool WebSocket::valid() const {
    return false;
}

WebSocket::Connection WebSocket::connect() {
    return {-1, "This build has no WebSocket support"};
}

WebSocket::Message WebSocket::read(std::string&) {
    return Message::Closed;
}

bool WebSocket::send(std::string_view) {
    return false;
}

void WebSocket::close() {}

void WebSocket::interrupt() {}

bool WebSocket::open() const {
    return false;
}

#endif

}  // namespace Jetstream::Platform
