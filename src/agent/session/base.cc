#include "impl.hh"

#include "tools/random.hh"
#include "tools/text.hh"

#include <algorithm>
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
    pending.clear();
    steerable = false;
    context.reset();
    outputs = std::make_shared<Detail::OutputStore>();
    contextTokens = 0;
    imageLimit.reset();
    imageLimitKey.clear();
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

std::string Session::Impl::drain(const std::string& prompt) {
    std::string text;
    for (const auto& entry : pending) {
        text += (text.empty() ? "" : "\n\n") + entry.text;
    }
    if (!Text::IsBlank(prompt)) {
        text += (text.empty() ? "" : "\n\n") + prompt;
    }
    pending.clear();
    return text;
}

std::optional<std::string> Session::Impl::absorb(Target& target) {
    std::lock_guard<std::mutex> lock(mutex);
    if (!steerable || pending.empty() || !owns(target)) {
        return std::nullopt;
    }

    auto text = drain({});
    auto user = std::make_shared<Entry>(Entry{true, text, {}});
    if (messages[target.assistantIndex]->segments.empty()) {
        messages.insert(messages.begin() + static_cast<I64>(target.assistantIndex), std::move(user));
    } else {
        messages.push_back(std::move(user));
        messages.push_back(std::make_shared<Entry>());
    }
    target.assistantIndex = messages.size() - 1;
    ++revision;
    return text;
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

U64 Session::Impl::imageCap(const Target target, const U64 configured, const std::string& key) const {
    std::lock_guard<std::mutex> lock(mutex);
    if (generation != target.generation || !imageLimit || imageLimitKey != key) {
        return configured;
    }
    return std::min(configured, *imageLimit);
}

void Session::Impl::limitImages(const Target target, const U64 limit, const std::string& key) {
    std::lock_guard<std::mutex> lock(mutex);
    if (generation != target.generation) {
        return;
    }
    imageLimit = imageLimitKey == key ? std::min(limit, imageLimit.value_or(limit)) : limit;
    imageLimitKey = key;
}

Session::Session(LiveFactory live, AudioFactory audio)
    : impl(std::make_unique<Impl>(std::move(live), std::move(audio))) {}

Session::~Session() {
    shutdown();
}

bool Session::submit(const std::string& prompt, const Config& config) {
    if (impl->streaming.load()) {
        return false;
    }
    {
        std::lock_guard<std::mutex> lock(impl->mutex);
        if (Text::IsBlank(prompt) && impl->pending.empty()) {
            return false;
        }
    }

    setMode(Mode::Text);

    Impl::Target target;
    std::shared_ptr<const std::vector<Provider::Message>> context;
    std::string sessionId;
    std::string text;
    {
        std::lock_guard<std::mutex> lock(impl->mutex);
        if (impl->streaming.load() || impl->callLog) {
            return false;
        }
        text = impl->drain(prompt);
        if (Text::IsBlank(text)) {
            return false;
        }
        impl->streaming.store(true);
        impl->steerable = true;
    }

    if (impl->worker.joinable()) {
        impl->worker.join();
    }

    {
        std::lock_guard<std::mutex> lock(impl->mutex);
        impl->messages.push_back(std::make_shared<Entry>(Entry{true, text, {}}));
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
                                text = std::move(text),
                                sessionId = std::move(sessionId),
                                config = config,
                                abort = impl->abortRequested,
                                outputs = impl->outputs]() mutable {
        while (true) {
            bool succeeded = false;
            {
                Impl::Turn turn(*impl, target, config, abort, outputs);
                turn.run(std::move(context), text, sessionId);
                succeeded = turn.succeeded();
            }

            std::unique_lock lock(impl->mutex);
            if (!succeeded || abort->load() || impl->pending.empty() || !impl->owns(target)) {
                impl->steerable = false;
                lock.unlock();
                config = {};
                lock.lock();
                impl->streaming.store(false);
                ++impl->revision;
                return;
            }

            text = impl->drain({});
            impl->messages.push_back(std::make_shared<Entry>(Entry{true, text, {}}));
            impl->messages.push_back(std::make_shared<Entry>());
            target.assistantIndex = impl->messages.size() - 1;
            context = impl->context;
            sessionId = impl->sessionId;
            ++impl->revision;
        }
    });

    return true;
}

bool Session::steer(const std::string& text) {
    if (Text::IsBlank(text)) {
        return false;
    }

    std::lock_guard<std::mutex> lock(impl->mutex);
    if (!impl->steerable || impl->callLog) {
        return false;
    }
    impl->pending.push_back({impl->nextPending++, text});
    ++impl->revision;
    return true;
}

bool Session::withdraw(U64 key) {
    std::lock_guard<std::mutex> lock(impl->mutex);
    const auto removed = std::erase_if(impl->pending, [key](const auto& entry) {
        return entry.key == key;
    });
    if (removed > 0) {
        ++impl->revision;
    }
    return removed > 0;
}

void Session::cancel() {
    impl->abortRequested->store(true);

    std::lock_guard<std::mutex> lock(impl->mutex);
    impl->steerable = false;
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
        .pending = impl->pending,
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
    update.pending = impl->pending;
    update.firstMessage = resume ? cursor.count - 1 : 0;

    update.messages.reserve(impl->messages.size() - update.firstMessage);
    for (U64 index = update.firstMessage; index < impl->messages.size(); ++index) {
        update.messages.push_back(*impl->messages[index]);
    }

    cursor = {impl->generation, impl->revision, impl->messages.size()};
    return update;
}

}  // namespace Jetstream::Agent
