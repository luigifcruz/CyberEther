#include <catch2/catch_test_macros.hpp>

#include "compositor/default/support/agent/runtime.hh"
#include "compositor/default/support/agent/providers.hh"
#include "compositor/default/model/callbacks.hh"
#include "compositor/default/model/state.hh"
#include "compositor/default/presenters/agent/transcript.hh"

#include "agent_voice.hh"

#include <atomic>
#include <chrono>
#include <future>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

using namespace Jetstream;
using namespace std::chrono_literals;

namespace {

using Phase = Agent::Session::Phase;

struct Harness {
    DefaultCompositorState state;
    DefaultCompositorCallbacks callbacks;
    std::mutex mutex;
    std::vector<std::string> notices;
    AgentRuntime runtime;

    explicit Harness(Agent::Session::LiveFactory factory);

    std::vector<std::string> recorded() {
        std::lock_guard lock(mutex);
        return notices;
    }

    Phase phase() const {
        return state.agent.voice.phase;
    }

    void dismiss() {
        runtime.hangup();
        runtime.poll();
    }
};

DefaultCompositorCallbacks Recording(Harness& harness) {
    DefaultCompositorCallbacks callbacks;
    callbacks.notify = [&harness](Sakura::ToastType, I32, const std::string& message) {
        std::lock_guard lock(harness.mutex);
        harness.notices.push_back(message);
    };
    return callbacks;
}

struct FakeConnection : Agent::Live {
    explicit FakeConnection(const Agent::Live::Callbacks& callbacks) : Live({}, callbacks) {}

    void send(const std::vector<I16>&) override {}

    void reply(const Agent::Live::ToolOutputs&) override {}
};

struct SilentAudio final : Agent::Session::Audio {
    SilentAudio() : Audio(24000) {}

    Result start() override {
        return Result::SUCCESS;
    }

    void stop() override {}

    std::vector<I16> capture(U64) override {
        return {};
    }

    void play(const std::string&) override {}

    void flush() override {}

    F32 input() const override {
        return 0.1f;
    }

    F32 output() const override {
        return 0.2f;
    }
};

std::unique_ptr<Agent::Session::Audio> Silent(U32) {
    return std::make_unique<SilentAudio>();
}

Harness::Harness(Agent::Session::LiveFactory factory) :
    callbacks(Recording(*this)),
    runtime(state, callbacks, {.live = std::move(factory), .audio = Silent}) {}

struct Gate {
    std::promise<void> entered;
    std::future<void> arrival = entered.get_future();
    std::promise<void> release;
    std::shared_future<void> released = release.get_future();

    void wait() {
        entered.set_value();
        released.wait();
    }
};

struct Connection final : FakeConnection {
    Agent::Live::Callbacks callbacks;
    Gate* starting;
    Gate* stopping;
    std::atomic<U64>& destroyed;

    Connection(Agent::Live::Callbacks callbacks, Gate* starting, Gate* stopping, std::atomic<U64>& destroyed)
        : FakeConnection(callbacks),
          callbacks(std::move(callbacks)),
          starting(starting),
          stopping(stopping),
          destroyed(destroyed) {}

    Result start() override {
        if (starting) {
            starting->wait();
        }

        callbacks.onState(Agent::Live::State::Listening, "");
        callbacks.onInputTranscript("late speech");
        return Result::SUCCESS;
    }

    void stop() override {
        if (stopping) {
            stopping->wait();
        }
    }

    ~Connection() override {
        ++destroyed;
    }
};

struct ClosingConnection final : FakeConnection {
    Agent::Live::Callbacks callbacks;
    Gate& teardown;

    ClosingConnection(Agent::Live::Callbacks callbacks, Gate& teardown)
        : FakeConnection(callbacks), callbacks(std::move(callbacks)), teardown(teardown) {}

    Result start() override {
        callbacks.onState(Agent::Live::State::Closed, "");
        return Result::SUCCESS;
    }

    void stop() override {
        teardown.wait();
        callbacks.onError("late teardown failure");
    }
};

}  // namespace

TEST_CASE("Voice poll progresses during connection and teardown and orders rapid restarts",
          "[agent][compositor][responsiveness]") {
    Gate starting, stopping;
    std::atomic<U64> created{0}, destroyed{0};
    std::promise<void> connectedPolls, dismissedPolls, dismiss, shutdown, shutdownRequested;
    auto connected = connectedPolls.get_future();
    auto dismissed = dismissedPolls.get_future();
    auto dismissNow = dismiss.get_future();
    auto shutdownNow = shutdown.get_future();
    auto shutdownStarted = shutdownRequested.get_future();
    bool connecting = true;
    U64 messagesAfterCancel = 0;

    auto ui = std::async(std::launch::async, [&] {
        Harness harness([&](auto, auto callbacks) {
            const auto index = created++;
            Gate* startGate = index == 0 ? &starting : nullptr;
            Gate* stopGate = index == 0 ? &stopping : nullptr;
            return std::make_unique<Connection>(std::move(callbacks), startGate, stopGate, destroyed);
        });
        auto& runtime = harness.runtime;

        runtime.listen({});
        for (int frame = 0; frame < 64; ++frame) {
            runtime.poll();
            connecting &= harness.phase() == Phase::Connecting;
        }
        connectedPolls.set_value();

        dismissNow.wait();
        harness.dismiss();
        runtime.listen({});
        for (int frame = 0; frame < 64; ++frame) {
            runtime.poll();
        }
        dismissedPolls.set_value();

        shutdownNow.wait();
        runtime.hangup();
        shutdownRequested.set_value();
        runtime.shutdown();
        messagesAfterCancel = runtime.session().snapshot().messages.size();
    });

    const bool entered = starting.arrival.wait_for(3s) == std::future_status::ready;
    const bool progressed = connected.wait_for(3s) == std::future_status::ready;

    dismiss.set_value();
    const bool dismissedPromptly = dismissed.wait_for(3s) == std::future_status::ready;
    const auto beforeRelease = created.load();

    starting.release.set_value();
    const bool teardownStarted = stopping.arrival.wait_for(3s) == std::future_status::ready;
    const auto whileStopping = created.load();

    shutdown.set_value();
    const bool stoppedBeforeRestart = shutdownStarted.wait_for(3s) == std::future_status::ready;
    stopping.release.set_value();
    ui.get();

    CHECK(entered);
    CHECK(progressed);
    CHECK(connecting);
    CHECK(dismissedPromptly);
    CHECK(beforeRelease == 1);
    CHECK(teardownStarted);
    CHECK(whileStopping == 1);
    CHECK(stoppedBeforeRestart);
    CHECK(destroyed == created);
    CHECK(messagesAfterCancel == 0);
}

TEST_CASE("Voice failures are notified by poll and release the conversation", "[agent][compositor][responsiveness]") {
    struct Failure final : FakeConnection {
        Agent::Live::Callbacks callbacks;
        Gate& gate;

        Failure(Agent::Live::Callbacks callbacks, Gate& gate)
            : FakeConnection(callbacks), callbacks(std::move(callbacks)), gate(gate) {}

        Result start() override {
            callbacks.onError("delayed connection failure");
            gate.wait();
            return Result::ERROR;
        }

        void stop() override {}
    };

    Gate gate;
    Harness harness([&](auto, auto callbacks) {
        return std::make_unique<Failure>(std::move(callbacks), gate);
    });
    auto& runtime = harness.runtime;

    runtime.listen({});
    const bool entered = gate.arrival.wait_for(3s) == std::future_status::ready;
    const auto beforePoll = harness.recorded().size();
    runtime.poll();
    const auto afterPoll = harness.recorded().size();

    gate.release.set_value();
    const auto deadline = std::chrono::steady_clock::now() + 3s;
    do {
        runtime.poll();
        std::this_thread::yield();
    } while (harness.phase() != Phase::Idle && std::chrono::steady_clock::now() < deadline);

    CHECK(entered);
    CHECK(beforePoll == 0);
    CHECK(afterPoll == 1);
    CHECK(harness.phase() == Phase::Idle);
    runtime.shutdown();
}

TEST_CASE("Chat snapshots reuse completed history while the tail streams",
          "[agent][compositor][responsiveness]") {
    AgentTest::VoiceLink link;
    Harness harness(link.factory());
    auto& session = harness.runtime.session();
    AgentTranscriptFormatter formatter;
    const auto snapshot = [&] {
        harness.runtime.poll();
        return formatter.format(harness.state.agent.chat.conversation);
    };

    harness.runtime.listen({});
    const auto voice = link.connection(0);
    REQUIRE(voice);
    for (int i = 0; i < 128; ++i) {
        voice->onInputTranscript(std::string(4096, 'x'));
        voice->onInputTranscriptDone();
    }
    voice->onOutputTranscript("first");

    const auto before = snapshot();
    REQUIRE(before.messages.size() == 129);

    const auto idle = snapshot();
    CHECK(idle.messages == before.messages);

    voice->onOutputTranscript(" second");
    const auto after = snapshot();
    CHECK(after.messages.history == before.messages.history);
    CHECK(after.messages.tail != before.messages.tail);
    CHECK(before.messages.tail->parts[0].text == "first");
    CHECK(after.messages.tail->parts[0].text == "first second");

    session.hangup();
    session.clear();
    CHECK(snapshot().messages.empty());
}

TEST_CASE("Voice closure retains terminal errors until teardown completion", "[agent][compositor][responsiveness]") {
    Gate teardown;
    Harness harness([&](auto, auto callbacks) {
        return std::make_unique<ClosingConnection>(std::move(callbacks), teardown);
    });
    auto& runtime = harness.runtime;

    runtime.listen({});
    const bool closing = teardown.arrival.wait_for(3s) == std::future_status::ready;
    for (int frame = 0; frame < 64; ++frame) {
        runtime.poll();
    }
    const auto beforeCompletion = harness.recorded().size();

    teardown.release.set_value();
    const auto deadline = std::chrono::steady_clock::now() + 3s;
    while (harness.recorded().empty() && std::chrono::steady_clock::now() < deadline) {
        runtime.poll();
        std::this_thread::yield();
    }
    const bool completed = !harness.recorded().empty();
    runtime.poll();
    runtime.shutdown();

    const auto notices = harness.recorded();
    CHECK(closing);
    CHECK(beforeCompletion == 0);
    CHECK(completed);
    REQUIRE(notices.size() == 1);
    CHECK(notices.back() == "late teardown failure");
}

TEST_CASE("Dismissed voice errors cannot leak into a subsequent session", "[agent][compositor][responsiveness]") {
    Gate teardown, starting;
    std::atomic<U64> created{0}, destroyed{0};
    Harness harness([&](auto, auto callbacks) -> std::unique_ptr<Agent::Live> {
        if (created++ == 0) {
            return std::make_unique<ClosingConnection>(std::move(callbacks), teardown);
        }
        return std::make_unique<Connection>(std::move(callbacks), &starting, nullptr, destroyed);
    });
    auto& runtime = harness.runtime;

    runtime.listen({});
    const bool closing = teardown.arrival.wait_for(3s) == std::future_status::ready;
    harness.dismiss();
    runtime.listen({});

    teardown.release.set_value();
    const bool restarted = starting.arrival.wait_for(3s) == std::future_status::ready;
    for (int frame = 0; frame < 64; ++frame) {
        runtime.poll();
    }
    const bool active = harness.phase() != Phase::Idle;

    starting.release.set_value();
    runtime.shutdown();

    CHECK(closing);
    CHECK(restarted);
    CHECK(active);
    CHECK(created == 2);
    CHECK(harness.recorded().empty());
}

TEST_CASE("Voice worker starts with fresh instructions and no conversation handoff", "[agent][compositor][history]") {
    struct PreparedConnection final : FakeConnection {
        Agent::Live::Callbacks callbacks;
        std::promise<void>& started;
        std::vector<std::string> speech;

        PreparedConnection(Agent::Live::Callbacks callbacks, std::promise<void>& started,
                           std::vector<std::string> speech)
            : FakeConnection(callbacks), callbacks(std::move(callbacks)), started(started), speech(std::move(speech)) {}

        Result start() override {
            callbacks.onState(Agent::Live::State::Listening, "");
            for (const auto& utterance : speech) {
                callbacks.onInputTranscript(utterance);
                callbacks.onInputTranscriptDone();
            }
            started.set_value();
            return Result::SUCCESS;
        }

        void stop() override {}
    };

    Agent::Live::Config prepared;
    std::thread::id preparationThread;
    std::promise<void> seeded;
    std::promise<void> started;
    auto seedReady = seeded.get_future();
    auto ready = started.get_future();
    U64 connections = 0;
    Harness harness([&](auto config, auto callbacks) {
        if (connections++ == 0) {
            return std::make_unique<PreparedConnection>(std::move(callbacks), seeded,
                                                        std::vector<std::string>{"Earlier question"});
        }
        prepared = std::move(config);
        preparationThread = std::this_thread::get_id();
        return std::make_unique<PreparedConnection>(std::move(callbacks), started,
                                                    std::vector<std::string>{"Current speech"});
    });
    auto& runtime = harness.runtime;
    auto& session = runtime.session();
    runtime.listen({});
    REQUIRE(seedReady.wait_for(3s) == std::future_status::ready);
    session.hangup();

    runtime.listen({});
    const bool connected = ready.wait_for(3s) == std::future_status::ready;
    runtime.shutdown();

    REQUIRE(connected);
    CHECK(preparationThread != std::this_thread::get_id());
    CHECK(prepared.instructions == AgentRuntime::VoicePrompt());
    CHECK(prepared.backendInstructions == AgentRuntime::SystemPrompt());

    const auto transcript = session.snapshot();
    REQUIRE_FALSE(transcript.messages.empty());
    CHECK(transcript.messages.back().text == "Current speech");
}
