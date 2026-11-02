#include "impl.hh"

#include "tools/random.hh"

#include <utility>

namespace Jetstream::Agent {

Session::Impl::Impl(LiveFactory live, AudioFactory audio)
    : sessionId(Random::HexToken()),
      liveFactory(std::move(live)),
      audioFactory(std::move(audio)),
      connector([this] {
          serve();
      }) {}

void Session::Impl::setMode(const Mode next) {
    if (mode == next) {
        return;
    }

    if (mode) {
        abortRequested->store(true);
        if (callLog) {
            retire();
        }
        reset();
    }

    mode = next;
}

void Session::Impl::reset() {
    ++generation;
    messages.clear();
    context.reset();
    outputs = std::make_shared<Detail::OutputStore>();
    contextTokens = 0;
    sessionId = Random::HexToken();
    ++revision;
}

void Session::Impl::retire() {
    callLog->cancelled->store(true);
    callLog.reset();
}

bool Session::Impl::owns(const Target target) const {
    return generation == target.generation && target.assistantIndex < messages.size();
}

void Session::Impl::append(const Target target, const Segment::Kind kind, const std::string& chunk) {
    std::lock_guard<std::mutex> lock(mutex);
    if (!owns(target)) {
        return;
    }

    auto& segments = messages[target.assistantIndex]->segments;
    if (segments.empty() || segments.back().kind != kind) {
        segments.push_back({kind, ""});
    }

    segments.back().text += chunk;
    segments.back().revision++;
    ++revision;
}

void Session::Impl::add(const Target target, Segment segment) {
    std::lock_guard<std::mutex> lock(mutex);
    if (!owns(target)) {
        return;
    }

    messages[target.assistantIndex]->segments.push_back(std::move(segment));
    ++revision;
}

void Session::Impl::record(const Target target, const Provider::Usage& usage) {
    const U64 tokens = usage.inputTokens + usage.outputTokens;

    std::lock_guard<std::mutex> lock(mutex);
    if (generation == target.generation && tokens > 0) {
        contextTokens = tokens;
        ++revision;
    }
}

void Session::Impl::publish(const Target target, const Turn& turn) {
    auto next = Detail::Context::Close(turn.history());
    if (!next.empty() && next.front().role == "system") {
        next.erase(next.begin());
    }
    auto retained = std::make_shared<const std::vector<Provider::Message>>(std::move(next));

    std::lock_guard<std::mutex> lock(mutex);
    if (owns(target)) {
        context = std::move(retained);
    }
}

Session::Session(LiveFactory live, AudioFactory audio)
    : impl(std::make_unique<Impl>(std::move(live), std::move(audio))) {}

Session::~Session() {
    shutdown();
}

bool Session::submit(const std::string& prompt, const Config& config) {
    if (prompt.find_first_not_of(" \t\r\n") == std::string::npos) {
        return false;
    }
    if (impl->streaming.load()) {
        return false;
    }

    setMode(Mode::Text);

    {
        std::lock_guard<std::mutex> lock(impl->mutex);
        if (impl->streaming.load() || impl->callLog) {
            return false;
        }
        impl->streaming.store(true);
    }

    if (impl->worker.joinable()) {
        impl->worker.join();
    }

    Impl::Target target;
    std::shared_ptr<const std::vector<Provider::Message>> context;
    std::string sessionId;
    {
        std::lock_guard<std::mutex> lock(impl->mutex);
        impl->messages.push_back(std::make_shared<Entry>(Entry{true, prompt, {}}));
        context = impl->context;
        sessionId = impl->sessionId;
        impl->messages.push_back(std::make_shared<Entry>());
        target = {impl->messages.size() - 1, impl->generation};
        ++impl->revision;
    }

    impl->abortRequested = std::make_shared<std::atomic<bool>>(false);
    impl->worker = std::thread([this,
                                target,
                                context = std::move(context),
                                prompt,
                                sessionId = std::move(sessionId),
                                config = config,
                                abort = impl->abortRequested,
                                outputs = impl->outputs]() mutable {
        {
            Impl::Turn turn(*impl, target, std::move(config), std::move(abort), std::move(outputs));
            turn.run(std::move(context), prompt, sessionId);
        }

        std::lock_guard lock(impl->mutex);
        impl->streaming.store(false);
        ++impl->revision;
    });

    return true;
}

void Session::cancel() {
    impl->abortRequested->store(true);

    std::lock_guard<std::mutex> lock(impl->mutex);
    if (impl->callLog) {
        impl->callLog->cancelled->store(true);
    }
}

void Session::clear() {
    cancel();

    std::lock_guard<std::mutex> lock(impl->mutex);
    if (impl->callLog) {
        impl->retire();
    }
    impl->reset();
}

void Session::setMode(Mode mode) {
    std::lock_guard lock(impl->mutex);
    impl->setMode(mode);
}

void Session::shutdown() {
    impl->stop();
    cancel();

    if (impl->worker.joinable()) {
        impl->worker.join();
    }
}

Session::Snapshot Session::snapshot() const {
    std::lock_guard<std::mutex> lock(impl->mutex);
    Snapshot snapshot{
        .generation = impl->generation,
        .streaming = impl->streaming.load(),
        .contextTokens = impl->contextTokens,
    };

    snapshot.messages.reserve(impl->messages.size());
    for (const auto& message : impl->messages) {
        snapshot.messages.push_back(*message);
    }

    return snapshot;
}

std::optional<Session::Update> Session::updates(Cursor& cursor) const {
    std::unique_lock lock(impl->mutex, std::try_to_lock);
    if (!lock.owns_lock() || cursor.revision == impl->revision) {
        return std::nullopt;
    }

    const bool resume = cursor.generation == impl->generation && cursor.count > 0;

    Update update;
    update.generation = impl->generation;
    update.streaming = impl->streaming.load();
    update.contextTokens = impl->contextTokens;
    update.firstMessage = resume ? cursor.count - 1 : 0;

    update.messages.reserve(impl->messages.size() - update.firstMessage);
    for (U64 index = update.firstMessage; index < impl->messages.size(); ++index) {
        update.messages.push_back(*impl->messages[index]);
    }

    cursor = {impl->generation, impl->revision, impl->messages.size()};
    return update;
}

}  // namespace Jetstream::Agent
