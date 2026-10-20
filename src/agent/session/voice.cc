#include "impl.hh"

#include "jetstream/config.hh"

#include <utility>

namespace Jetstream::Agent {

namespace {

constexpr U64 kChunkMilliseconds = 40;
constexpr auto kPumpInterval = std::chrono::milliseconds(10);

std::unique_ptr<Live> MakeLive(Live::Config config, Live::Callbacks callbacks) {
    return std::make_unique<Live>(std::move(config), std::move(callbacks));
}

std::unique_ptr<Session::Audio> MakeAudio([[maybe_unused]] U32 sampleRate) {
#if defined(JETSTREAM_AGENT_LIVE_AVAILABLE)
    return std::make_unique<Session::Audio>(sampleRate);
#else
    return nullptr;
#endif
}

}  // namespace

std::optional<U64> Session::Impl::beginVoice(std::shared_ptr<std::atomic<bool>> cancelled) {
    std::lock_guard<std::mutex> lock(mutex);
    setMode(Mode::Voice);
    if (callLog) {
        return std::nullopt;
    }

    const U64 id = ++nextVoiceId;
    callLog.emplace(CallLog{
        .id = id,
        .cancelled = std::move(cancelled),
    });

    return id;
}

void Session::Impl::appendVoiceTranscript(U64 id, bool user, const std::string& delta) {
    if (delta.empty()) {
        return;
    }

    std::lock_guard<std::mutex> lock(mutex);
    auto* active = activeLog(id);
    if (!active) {
        return;
    }

    auto& run = *active;
    if (!run.openMessage || run.openUser != user) {
        run.openMessage = messages.size();
        run.openUser = user;
        messages.push_back(std::make_shared<Entry>(Entry{user, "", {}}));
    }

    auto& message = *messages[*run.openMessage];
    if (user) {
        message.text += delta;
    } else {
        if (message.segments.empty()) {
            message.segments.push_back({Segment::Kind::Text, ""});
        }
        message.segments.back().text += delta;
        message.segments.back().revision++;
    }

    ++revision;
}

void Session::Impl::finishVoiceTranscript(U64 id, bool user) {
    std::lock_guard<std::mutex> lock(mutex);
    if (auto* run = findLog(id); run && run->openUser == user) {
        run->openMessage.reset();
    }
}

void Session::Impl::finishVoiceTool(U64 id) {
    std::lock_guard<std::mutex> lock(mutex);
    if (auto* active = activeLog(id)) {
        active->openMessage.reset();
    }
}

void Session::Impl::endVoice(U64 id) {
    std::lock_guard<std::mutex> lock(mutex);
    if (findLog(id)) {
        retire();
    }
}

bool Session::Impl::voiceActive() const {
    std::lock_guard<std::mutex> lock(mutex);
    return callLog && !callLog->cancelled->load();
}

Session::Impl::CallLog* Session::Impl::findLog(const U64 id) {
    return callLog && callLog->id == id ? &*callLog : nullptr;
}

Session::Impl::CallLog* Session::Impl::activeLog(const U64 id) {
    auto* run = findLog(id);
    if (!run || run->cancelled->load()) {
        return nullptr;
    }
    return run;
}

void Session::Impl::Call::close() {
    {
        std::lock_guard lock(mutex);
        closing.store(true);
    }
    changed.notify_all();
}

Live::Callbacks Session::Impl::callbacks(const std::shared_ptr<Call>& owner) {
    Live::Callbacks callbacks;
    const std::weak_ptr<Call> weak = owner;
    const U64 id = owner->id;

    callbacks.onState = [weak](const Live::State next, const std::string&) {
        if (const auto call = weak.lock()) {
            std::lock_guard lock(call->mutex);
            call->state.store(next);
            call->changed.notify_all();
        }
    };

    callbacks.onDelegation = [weak](const bool active) {
        if (const auto call = weak.lock()) {
            call->delegating.store(active);
        }
    };

    callbacks.onError = [weak](const std::string& message) {
        const auto call = weak.lock();
        if (!call) {
            return;
        }
        std::lock_guard lock(call->mutex);
        if (call->stopping) {
            return;
        }
        call->errors.push_back(message);
    };

    callbacks.onInputTranscript = [this, id](const std::string& delta) {
        appendVoiceTranscript(id, true, delta);
    };

    callbacks.onOutputTranscript = [this, id](const std::string& delta) {
        appendVoiceTranscript(id, false, delta);
    };

    callbacks.onInputTranscriptDone = [this, id, weak]() {
        if (const auto call = weak.lock()) {
            call->tools->reset();
        }
        finishVoiceTranscript(id, true);
    };

    callbacks.onOutputTranscriptDone = [this, id]() {
        finishVoiceTranscript(id, false);
    };

    callbacks.onAudio = [weak](const std::string& base64) {
        if (const auto call = weak.lock(); call && call->audio) {
            call->audio->play(base64);
        }
    };

    callbacks.onAudioClear = [weak]() {
        if (const auto call = weak.lock(); call && call->audio) {
            call->audio->flush();
        }
    };

    callbacks.onFunctionCalls = [weak](std::vector<Provider::FunctionCall> batch) {
        const auto call = weak.lock();
        if (!call) {
            return;
        }
        {
            std::lock_guard lock(call->mutex);
            call->batches.push_back(std::move(batch));
        }
        call->changed.notify_all();
    };

    return callbacks;
}

void Session::Impl::pump(Call& call) {
    const U64 chunk = static_cast<U64>(call.config.sampleRate) * kChunkMilliseconds / 1000;

    std::unique_lock lock(call.mutex);
    while (!call.closing.load()) {
        lock.unlock();

        call.input.store(call.audio->input());
        call.output.store(call.audio->output());

        if (call.state.load() == Live::State::Listening) {
            const auto samples = call.audio->capture(chunk);
            if (!samples.empty()) {
                call.live->send(samples);
            }
        }

        lock.lock();
        call.changed.wait_for(lock, kPumpInterval, [&] {
            return call.closing.load();
        });
    }
}

void Session::Impl::runTools(Call& call) {
    while (true) {
        std::vector<Provider::FunctionCall> batch;
        {
            std::unique_lock lock(call.mutex);
            call.changed.wait(lock, [&] {
                return call.closing.load() || !call.batches.empty();
            });

            if (call.closing.load()) {
                return;
            }
            batch = std::move(call.batches.front());
            call.batches.pop_front();
        }

        call.delegating.store(true);
        const bool lastRound = call.tools->advance();

        try {
            Live::ToolOutputs results;
            for (const auto& request : batch) {
                Tools::Output result;
                try {
                    result = call.tools->run(request, call.resultLimit, lastRound, call.aborted);
                } catch (const std::exception& error) {
                    result = Tools::Error(std::string("The tool failed: ") + error.what(), true);
                } catch (...) {
                    result = Tools::Error("The tool failed with an unknown error.", true);
                }
                if (call.closing.load()) {
                    return;
                }

                finishVoiceTool(call.id);
                results.emplace_back(request, std::move(result));
            }

            call.live->reply(results, call.tools->exhausted());
        } catch (const std::exception& error) {
            call.callbacks.onError(std::string("The voice tool call failed: ") + error.what());
        } catch (...) {
            call.callbacks.onError("The voice tool call failed unexpectedly.");
        }
    }
}

void Session::Impl::connect(Call& call) {
    if (call.cancelled->load()) {
        return;
    }

    const U32 sampleRate = call.config.sampleRate;
    call.audio = audioFactory ? audioFactory(sampleRate) : MakeAudio(sampleRate);
    auto live = liveFactory ? liveFactory(call.config, call.callbacks) : MakeLive(call.config, call.callbacks);
    call.live = live.get();

    if (live->start() != Result::SUCCESS) {
        live->stop();
        return;
    }
    if (call.cancelled->load()) {
        live->stop();
        return;
    }

    if (!call.audio || call.audio->start() != Result::SUCCESS) {
        call.callbacks.onError("Could not start the microphone and speaker for the voice session.");
        live->stop();
        return;
    }

    std::thread tools([&] {
        runTools(call);
    });
    std::thread audio([&] {
        pump(call);
    });

    {
        std::unique_lock lock(call.mutex);
        call.changed.wait(lock, [&] {
            return call.stopping || call.state.load() == Live::State::Closed;
        });
    }

    call.close();
    audio.join();
    call.audio->stop();
    tools.join();
    live->stop();
    call.live = nullptr;
}

void Session::Impl::serve() {
    while (true) {
        std::shared_ptr<Call> next;
        {
            std::unique_lock lock(callMutex);
            callChanged.wait(lock, [&] {
                return quitting || !calls.empty();
            });

            if (calls.empty()) {
                return;
            }
            next = std::move(calls.front());
            calls.pop_front();
        }

        {
            std::lock_guard lock(next->mutex);
            if (next->stopping) {
                next->finished = true;
                continue;
            }
        }

        try {
            connect(*next);
        } catch (const std::exception& error) {
            next->callbacks.onError(error.what());
        } catch (...) {
            next->callbacks.onError("The voice session failed unexpectedly.");
        }

        {
            std::lock_guard lock(next->mutex);
            next->finished = true;
        }
    }
}

void Session::Impl::disconnect() {
    if (!call) {
        return;
    }

    endVoice(call->id);

    {
        std::lock_guard lock(call->mutex);
        call->stopping = true;
    }

    call->changed.notify_all();
    call.reset();
}

void Session::Impl::stop() {
    disconnect();

    {
        std::lock_guard lock(callMutex);
        quitting = true;
    }
    callChanged.notify_all();

    if (connector.joinable()) {
        connector.join();
    }
}

bool Session::listen(const Config& config) {
    if (impl->call || impl->quitting) {
        return false;
    }

    auto call = std::make_shared<Impl::Call>();
    call->config = {
        .provider = config.provider,
        .model = config.voice.model,
        .backendModel = config.model,
        .voice = config.voice.name,
        .instructions = config.voice.instructions,
        .backendInstructions = config.systemContext,
        .tools = config.tools,
    };
    call->config.tools.push_back(Detail::OutputStore::Schema());
    call->cancelled = std::make_shared<std::atomic<bool>>(false);
    call->aborted = [flag = call->cancelled, weak = std::weak_ptr<Impl::Call>(call)]() {
        const auto alive = weak.lock();
        return flag->load() || !alive || alive->closing.load();
    };

    const auto id = impl->beginVoice(call->cancelled);
    if (!id) {
        return false;
    }

    call->id = *id;
    call->tools = std::make_unique<Detail::ToolRunner>(config.executeTool,
                                                       std::make_shared<Detail::OutputStore>(),
                                                       config.maxToolTurns);
    call->resultLimit = config.maxToolResultChars;
    call->callbacks = impl->callbacks(call);
    impl->call = call;

    {
        std::lock_guard lock(impl->callMutex);
        impl->calls.push_back(std::move(call));
    }

    impl->callChanged.notify_one();
    return true;
}

void Session::hangup() {
    impl->disconnect();
}

std::vector<std::string> Session::poll() {
    std::vector<std::string> errors;
    const auto call = impl->call;
    if (!call) {
        return errors;
    }

    bool finished;
    {
        std::lock_guard lock(call->mutex);
        errors.swap(call->errors);
        finished = call->finished;
    }

    if (finished || !impl->voiceActive()) {
        impl->disconnect();
    }

    return errors;
}

Session::Status Session::status() const {
    Status snapshot;
    const auto& call = impl->call;
    if (!call) {
        return snapshot;
    }

    switch (call->state.load()) {
        case Live::State::Listening:
            snapshot.phase = call->delegating.load() ? Phase::Delegating : Phase::Listening;
            break;
        case Live::State::Idle:
        case Live::State::Connecting:
            snapshot.phase = Phase::Connecting;
            break;
        case Live::State::Closed:
            return snapshot;
    }

    snapshot.input = call->input.load();
    snapshot.output = call->output.load();
    return snapshot;
}

}  // namespace Jetstream::Agent
