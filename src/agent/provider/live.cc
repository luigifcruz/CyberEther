#include "jetstream/agent/live.hh"

#include "jetstream/config.hh"
#include "jetstream/logger.hh"
#include "jetstream/platform.hh"

#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <exception>
#include <mutex>
#include <thread>
#include <utility>

#include "support/base.hh"
#include "tools/base64.hh"

namespace Jetstream::Agent {

std::span<const std::string> Live::Models() {
    static const std::array<std::string, 1> models{
        "gpt-live-1",
    };
    return models;
}

std::span<const std::string> Live::Voices() {
    static const std::array<std::string, 10> voices{
        "marin",
        "cedar",
        "alloy",
        "ash",
        "ballad",
        "coral",
        "echo",
        "sage",
        "shimmer",
        "verse",
    };
    return voices;
}

namespace {

constexpr auto kCloseGrace = std::chrono::milliseconds(750);
constexpr auto kSocketCloseGrace = std::chrono::milliseconds(1000);
constexpr U64 kPingIntervalSeconds = 20;
constexpr auto kIdleWait = std::chrono::milliseconds(10);

std::string EventError(const nlohmann::json& event) {
    auto message = Detail::Wire::Error(event);
    if (message.empty()) {
        return event.dump();
    }
    return message;
}

std::string TerminalType(const nlohmann::json& event) {
    const auto response = event.find("response");
    const std::string status = response != event.end() && response->is_object()
                                   ? Detail::Wire::Field(*response, "status")
                                   : std::string{};
    if (status == "incomplete") {
        return "response.incomplete";
    }
    if (status == "failed" || status == "cancelled" || Detail::Responses::Failed(event)) {
        return "response.failed";
    }
    return "response.completed";
}

std::string SocketUrl(std::string host, const std::string& path) {
    if (host.starts_with("https://")) {
        host.replace(0, 5, "wss");
    } else if (host.starts_with("http://")) {
        host.replace(0, 4, "ws");
    }

    while (host.ends_with('/')) {
        host.pop_back();
    }
    return host + path;
}

}  // namespace

struct Live::Impl {
    Config config;
    Callbacks callbacks;
    std::string key;

    std::atomic<State> state{State::Idle};
    std::atomic<bool> active{false};
    std::atomic<bool> connected{false};
    std::atomic<bool> closing{false};
    std::atomic<bool> sessionClosed{false};
    std::atomic<bool> delegating{false};
    std::atomic<bool> readerDone{false};
    std::atomic<bool> orphaned{false};
    std::atomic<std::thread::id> ioId{};
    std::atomic<std::thread::id> readerId{};
    std::unique_ptr<Platform::WebSocket> socket;
    std::mutex outboundMutex;
    std::vector<std::string> queuedOutbound;

    std::atomic<U64> eventCounter{0};
    std::atomic<U64> receivedEvents{0};
    std::atomic<U64> sentEvents{0};
    std::string lastEventLabel;
    std::chrono::steady_clock::time_point startedAt;

    std::unique_ptr<Detail::EventStream> response;
    std::thread ioThread;

    std::mutex closeMutex;
    std::condition_variable closeCv;

    std::mutex launchMutex;
    std::condition_variable launchCv;
    bool launched = false;

    ~Impl() {
        closing.store(true);
        if (ioThread.joinable()) {
            ioThread.join();
        }
    }

    std::string host() const {
        return config.provider ? config.provider->config().host : std::string{};
    }

    std::string path() const {
        return config.provider ? config.provider->profile().livePath : std::string{};
    }

    std::string redact(std::string text) const {
        return Detail::Diagnostics::Redact(std::move(text), key);
    }

    std::string preview(const std::string& payload) const {
        auto redacted = redact(payload);
        if (redacted.size() > 2048) {
            return redacted.substr(0, 2048) + "...";
        }
        return redacted;
    }

    template <typename Callback, typename... Args>
    void emit(const Callback& callback, Args&&... args) {
        if (callback && !orphaned.load()) {
            callback(std::forward<Args>(args)...);
        }
    }

    bool owned() const {
        const auto current = std::this_thread::get_id();
        return current == ioId.load() || current == readerId.load();
    }

    void setState(const State next, const std::string& detail = "") {
        state.store(next);
        emit(callbacks.onState, next, detail);
    }

    void reportError(const std::string& message) {
        const auto redacted = redact(message);
        JST_ERROR("[LIVE] {}", redacted);
        emit(callbacks.onError, redacted);
    }

    void markSessionClosed() {
        {
            std::lock_guard<std::mutex> lock(closeMutex);
            sessionClosed.store(true);
        }
        closeCv.notify_all();
    }

    bool queue(std::string payload) {
        if (!connected.load() || closing.load()) {
            return false;
        }

        std::lock_guard<std::mutex> lock(outboundMutex);
        queuedOutbound.push_back(std::move(payload));
        return true;
    }

    bool sendEvent(nlohmann::json event) {
        event["event_id"] = jst::fmt::format("evt_{}", eventCounter.fetch_add(1));
        const std::string type = event.value("type", "");
        const std::string payload = event.dump();

        if (!queue(payload)) {
            JST_WARN("[LIVE] Dropping '{}' because the session is not open.", type);
            return false;
        }

        JST_INFO("[LIVE] Sending: {}", type);
        JST_DEBUG("[LIVE] Payload: {}", preview(payload));
        sentEvents.fetch_add(1);
        return true;
    }

    std::string describeTransportFailure() const {
        const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - startedAt).count();
        return jst::fmt::format("after {} ms, {} events received, {} sent, last event '{}'",
                                elapsed, receivedEvents.load(), sentEvents.load(), lastEventLabel);
    }

    void setDelegating(const bool next) {
        if (delegating.exchange(next) == next) {
            return;
        }

        emit(callbacks.onDelegation, next);
    }

    nlohmann::json sessionStartEvent() const {
        nlohmann::json tools = nlohmann::json::array();
        for (const auto& tool : config.tools) {
            tools.push_back(Detail::Responses::Function(tool));
        }

        nlohmann::json responses;
        responses["model"] = config.backendModel;
        if (!config.backendInstructions.empty()) {
            responses["instructions"] = config.backendInstructions;
        }
        if (!tools.empty()) {
            responses["tools"] = std::move(tools);
            responses["tool_choice"] = "auto";
        }

        nlohmann::json session;
        session["model"] = config.model;
        if (!config.instructions.empty()) {
            session["instructions"] = config.instructions;
        }
        session["audio"] = {
            {"format", {{"type", "audio/pcm"}, {"rate", config.sampleRate}}},
            {"output", {{"voice", config.voice}}},
        };
        session["delegation"] = {
            {"type", "responses"},
            {"responses", std::move(responses)},
        };

        return {{"type", "session.start"}, {"session", std::move(session)}};
    }

    void finishResponse(const std::string& type, const bool decoded) {
        auto finished = std::move(response);
        setDelegating(false);
        if (!finished) {
            return;
        }

        if (type == "response.completed" && decoded && finished->settle()) {
            if (!finished->toolCalls.empty()) {
                emit(callbacks.onFunctionCalls, std::move(finished->toolCalls));
            }
            return;
        }

        const auto& message = finished->error.message;
        if (message.empty()) {
            reportError("The backend model failed to complete a delegated request.");
        } else {
            reportError(jst::fmt::format("The backend model failed: {}", message));
        }
    }

    void handleResponseEvent(const nlohmann::json& outer) {
        const auto inner = outer.find("event");
        if (inner == outer.end() || !inner->is_object()) {
            return;
        }

        const std::string type = inner->value("type", "");
        if (type == "error") {
            reportError(EventError(*inner));
            return;
        }

        if (type == "response.created") {
            response = Detail::Responses::Decoder();
        }
        if (type == "response.created" || type == "response.in_progress") {
            setDelegating(true);
        }
        if (!response) {
            response = Detail::Responses::Decoder();
        }

        std::string kind = type;
        bool decoded = false;
        if (type == "response.done") {
            auto terminal = *inner;
            kind = TerminalType(terminal);
            terminal["type"] = kind;
            decoded = response->feed(terminal);
        } else {
            decoded = response->feed(*inner);
        }

        const bool finished = kind == "response.completed" || kind == "response.incomplete" ||
                              kind == "response.failed";
        if (finished) {
            finishResponse(kind, decoded);
        }
    }

    void recordEvent(const nlohmann::json& event, const std::string& type, const std::string& payload) {
        receivedEvents.fetch_add(1);

        std::string label = type;
        if (type == "response.event") {
            const auto inner = event.find("event");
            if (inner != event.end() && inner->is_object()) {
                label += " / " + inner->value("type", "");
            }
        }
        lastEventLabel = label;

        const bool streamingDelta = type == "session.output_audio.delta" ||
                                    type == "session.input_transcript.delta" ||
                                    type == "session.output_transcript.delta";
        if (!streamingDelta) {
            JST_INFO("[LIVE] Event: {} ({} bytes)", redact(label), payload.size());
            JST_DEBUG("[LIVE] Payload: {}", preview(payload));
        }
    }

    void handleOutputAudio(const std::string& type, const nlohmann::json& event) {
        if (type == "session.output_audio.delta") {
            const auto delta = event.find("delta");
            if (delta != event.end() && delta->is_string()) {
                emit(callbacks.onAudio, delta->get<std::string>());
            }
            return;
        }

        const std::string tail = type.substr(std::strlen("session.output_audio."));
        const bool clears = tail.find("clear") != std::string::npos ||
                            tail.find("cancel") != std::string::npos ||
                            tail.find("interrupt") != std::string::npos ||
                            tail.find("truncat") != std::string::npos;
        if (clears) {
            emit(callbacks.onAudioClear);
        }
    }

    bool handleTranscriptDone(const std::string& type) {
        if (type != "session.input_transcript.done" && type != "session.output_transcript.done") {
            return false;
        }

        const bool input = type == "session.input_transcript.done";
        emit(input ? callbacks.onInputTranscriptDone : callbacks.onOutputTranscriptDone);
        return true;
    }

    bool handleTranscriptDelta(const std::string& type, const nlohmann::json& event) {
        if (type != "session.input_transcript.delta" && type != "session.output_transcript.delta") {
            return false;
        }

        const auto delta = event.find("delta");
        if (delta == event.end() || !delta->is_string()) {
            return true;
        }

        const bool input = type == "session.input_transcript.delta";
        emit(input ? callbacks.onInputTranscript : callbacks.onOutputTranscript, delta->get<std::string>());
        return true;
    }

    void handleSessionEvent(const std::string& type, const nlohmann::json& event) {
        if (type == "session.started") {
            JST_INFO("[LIVE] Voice session started with '{}' delegating to '{}'.",
                     config.model, config.backendModel);
            setState(State::Listening);
            return;
        }

        if (type == "session.closed") {
            JST_INFO("[LIVE] Voice session closed ({}).", redact(event.value("reason", "unspecified")));
            markSessionClosed();
            closing.store(true);
            return;
        }

        if (type == "error") {
            reportError(EventError(event));
            return;
        }

        JST_TRACE("[LIVE] Unhandled event '{}'.", type);
    }

    void handleEvent(const std::string& raw) {
        const auto payload = redact(raw);
        const auto event = nlohmann::json::parse(payload, nullptr, false);
        if (event.is_discarded() || !event.is_object()) {
            JST_WARN("[LIVE] Ignoring a malformed event from the server.");
            return;
        }

        const std::string type = event.value("type", "");
        recordEvent(event, type, payload);

        if (type.rfind("session.output_audio.", 0) == 0) {
            handleOutputAudio(type, event);
            return;
        }
        if (handleTranscriptDone(type) || handleTranscriptDelta(type, event)) {
            return;
        }
        if (type == "response.event") {
            handleResponseEvent(event);
            return;
        }

        handleSessionEvent(type, event);
    }

    void readLoop() {
        readerId.store(std::this_thread::get_id());
        std::string message;
        while (!orphaned.load()) {
            const auto result = socket->read(message);
            if (result == Platform::WebSocket::Message::Closed) {
                break;
            }
            if (result != Platform::WebSocket::Message::Text) {
                continue;
            }

            try {
                handleEvent(message);
            } catch (const std::exception& e) {
                JST_WARN("[LIVE] Failed to handle a server event: {}", redact(e.what()));
            }
        }

        readerDone.store(true);
    }

    bool pumpFrames() {
        std::vector<std::string> frames;
        while (!readerDone.load()) {
            {
                std::lock_guard<std::mutex> lock(outboundMutex);
                frames.swap(queuedOutbound);
            }

            for (const auto& frame : frames) {
                if (!socket->send(frame)) {
                    return false;
                }
            }
            frames.clear();

            if (closing.load()) {
                break;
            }
            std::this_thread::sleep_for(kIdleWait);
        }

        return true;
    }

    void closeSocket(std::thread& reader, const bool writeFailed) {
        connected.store(false);
        if (!writeFailed) {
            socket->close();
        }

        const auto grace = writeFailed ? std::chrono::milliseconds(0) : kSocketCloseGrace;
        const auto deadline = std::chrono::steady_clock::now() + grace;
        while (!readerDone.load() && std::chrono::steady_clock::now() < deadline) {
            std::this_thread::sleep_for(kIdleWait);
        }

        socket->interrupt();
        reader.join();
        readerId.store({});
        socket.reset();
        setDelegating(false);
    }

    void finishSession(const bool writeFailed) {
        std::string reason;
        if (!closing.load() && !sessionClosed.load()) {
            const char* failure = writeFailed ? "Socket write failed" : "Socket read failed";
            JST_ERROR("[LIVE] {}: {}", failure, redact(describeTransportFailure()));
            reason = "The voice connection was lost.";
            reportError(reason);
        } else if (writeFailed) {
            JST_DEBUG("[LIVE] Socket write failed: {}", redact(describeTransportFailure()));
        }

        markSessionClosed();
        active.store(false);
        setState(State::Closed, reason);
    }

    void ioLoop() {
        {
            std::unique_lock lock(launchMutex);
            launchCv.wait(lock, [this] {
                return launched;
            });
        }
        ioId.store(std::this_thread::get_id());
        readerDone.store(false);
        std::thread reader([this]() { readLoop(); });
        const bool writeFailed = !pumpFrames();
        closeSocket(reader, writeFailed);
        finishSession(writeFailed);
        ioId.store({});
        if (orphaned.load()) {
            delete this;
        }
    }

    bool openSocket() {
        Platform::WebSocket::Config socketConfig;
        socketConfig.url = SocketUrl(host(), path());
#if defined(JST_OS_BROWSER)
        socketConfig.protocols = {"realtime", "openai-insecure-api-key." + key};
#else
        socketConfig.headers.emplace_back("Authorization", "Bearer " + key);
#endif
        socketConfig.connectTimeoutSeconds = 10;
        socketConfig.writeTimeoutSeconds = 30;
        socketConfig.pingIntervalSeconds = kPingIntervalSeconds;
        socketConfig.bufferBytes = 1 << 20;

        socket = std::make_unique<Platform::WebSocket>(std::move(socketConfig));
        if (!socket->valid()) {
            socket.reset();
            reportError(jst::fmt::format("Invalid voice session URL '{}{}'.", host(), path()));
            return false;
        }
        return true;
    }

    void resetSession() {
        closing.store(false);
        sessionClosed.store(false);
        receivedEvents.store(0);
        sentEvents.store(0);
        lastEventLabel.clear();
        response.reset();
        startedAt = std::chrono::steady_clock::now();

        std::lock_guard<std::mutex> lock(outboundMutex);
        queuedOutbound.clear();
    }

    bool connectSocket() {
        if (const auto connection = socket->connect(); !connection) {
            std::string status;
            if (connection.status != -1) {
                status = jst::fmt::format(", HTTP {}", connection.status);
            }
            reportError(jst::fmt::format("Could not connect to '{}{}' ({}{}).",
                                         host(), path(), connection.error, status));
            socket.reset();
            setState(State::Idle);
            return false;
        }

        connected.store(true);
        sendEvent(sessionStartEvent());
        return true;
    }

    std::string audioAppendEvent(const std::vector<I16>& samples) {
        const auto* bytes = reinterpret_cast<const U8*>(samples.data());
        nlohmann::json event = {
            {"type", "session.input_audio.append"},
            {"event_id", jst::fmt::format("evt_{}", eventCounter.fetch_add(1))},
            {"audio", Base64::Encode(bytes, samples.size() * sizeof(I16))},
        };
        return event.dump();
    }
};

Live::Live(Config config, Callbacks callbacks) : impl(std::make_unique<Impl>()) {
    impl->config = std::move(config);
    impl->callbacks = std::move(callbacks);
}

Live::~Live() {
    if (impl->owned()) {
        impl->closing.store(true);
        impl->orphaned.store(true);
        impl->ioThread.detach();
        impl.release();
        return;
    }
    stop();
}

bool Live::Available() {
#if defined(JETSTREAM_AGENT_LIVE_AVAILABLE)
    return true;
#else
    return false;
#endif
}

Result Live::start() {
    if (impl->active.load()) {
        return Result::SUCCESS;
    }
    if (impl->owned()) {
        JST_ERROR("[LIVE] Cannot restart a voice session from its own callback.");
        return Result::ERROR;
    }

    stop();

    const auto& provider = impl->config.provider;
    impl->key = provider && provider->config().apiKey ? provider->config().apiKey() : std::string{};
    if (impl->key.empty()) {
        impl->reportError(Detail::Diagnostics::MissingKey());
        return Result::ERROR;
    }

    if (!impl->openSocket()) {
        return Result::ERROR;
    }

    impl->resetSession();
    impl->setState(State::Connecting);
    if (!impl->connectSocket()) {
        return Result::ERROR;
    }

    impl->active.store(true);
    {
        std::lock_guard lock(impl->launchMutex);
        impl->launched = false;
    }
    impl->ioThread = std::thread([state = impl.get()]() { state->ioLoop(); });
    {
        std::lock_guard lock(impl->launchMutex);
        impl->launched = true;
        impl->launchCv.notify_one();
    }
    return Result::SUCCESS;
}

void Live::stop() {
    if (!impl->ioThread.joinable()) {
        return;
    }

    const bool owned = impl->owned();
    if (!impl->sessionClosed.load() && impl->connected.load()) {
        impl->sendEvent({{"type", "session.close"}});
        if (!owned) {
            std::unique_lock<std::mutex> lock(impl->closeMutex);
            impl->closeCv.wait_for(lock, kCloseGrace, [this]() {
                return impl->sessionClosed.load();
            });
        }
    }

    impl->closing.store(true);
    if (owned) {
        return;
    }
    impl->ioThread.join();

    impl->active.store(false);
    if (impl->state.load() != State::Closed) {
        impl->setState(State::Closed);
    }
}

void Live::send(const std::vector<I16>& samples) {
    if (samples.empty() || impl->state.load() != State::Listening) {
        return;
    }

    impl->queue(impl->audioAppendEvent(samples));
}

void Live::reply(const ToolOutputs& outputs, bool disableTools) {
    nlohmann::json imageMessages = nlohmann::json::array();
    for (const auto& [call, result] : outputs) {
        impl->sendEvent({
            {"type", "response.item.create"},
            {"item", Detail::Responses::FunctionOutput(call.id, result.content)},
        });

        if (!result.images.empty()) {
            imageMessages.push_back({
                {"type", "message"},
                {"role", "user"},
                {"content", Detail::Responses::InputContent(call.caption(), result.images)},
            });
        }
    }

    for (auto& message : imageMessages) {
        impl->sendEvent({
            {"type", "response.item.create"},
            {"item", std::move(message)},
        });
    }

    if (disableTools) {
        impl->sendEvent({
            {"type", "response.create"},
            {"response", {{"tool_choice", "none"}}},
        });
    } else {
        impl->sendEvent({{"type", "response.create"}});
    }
}

}  // namespace Jetstream::Agent
