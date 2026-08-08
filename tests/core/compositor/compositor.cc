#include <catch2/catch_test_macros.hpp>

#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <functional>
#include <future>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "jetstream/compositor.hh"
#include "jetstream/detail/compositor_impl.hh"
#include "jetstream/logger.hh"

#include "compositor/default/actions/settings.hh"
#include "compositor/default/presenters/menubar.hh"

using namespace Jetstream;

namespace {

class CompositorWorkerDouble final : public Compositor::Impl {
 public:
    using CompletedCommand = Command;

    struct State {
        bool running;
        std::size_t pending;
        std::size_t completed;
    };

    ~CompositorWorkerDouble() override {
        stopWorker();
    }

    Result create() override {
        return Result::SUCCESS;
    }

    Result destroy() override {
        return Result::SUCCESS;
    }

    Result present() override {
        return Result::SUCCESS;
    }

    Result poll() override {
        return Result::SUCCESS;
    }

    void start() {
        startWorker();
    }

    void stop() {
        stopWorker();
    }

    void submit(std::function<Result()> fn, bool silent = false) {
        enqueue(std::move(fn), silent);
    }

    bool take(CompletedCommand& command) {
        return dequeue(command);
    }

    State state() {
        std::scoped_lock lock(commandPendingQueueMutex,
                              commandCompletedQueueMutex);
        return {
            .running = workerRunning,
            .pending = commandPendingQueue.size(),
            .completed = commandCompletedQueue.size(),
        };
    }

    bool waitUntilStoppedAccepting() {
        std::unique_lock lock(commandPendingQueueMutex);
        return commandQueueNotEmpty.wait_for(
            lock,
            std::chrono::seconds(2),
            [this] { return !workerRunning; });
    }
};

void RequireRejectedFactoryChoice(CompositorType type) {
    JST_LOG_LAST_FATAL().clear();

    try {
        Compositor compositor{type};
        FAIL("Unsupported compositor type did not throw.");
    } catch (const Result& result) {
        REQUIRE(result == Result::FATAL);
    }

    REQUIRE(JST_LOG_LAST_FATAL() == "[COMPOSITOR] Unknown compositor type.");
}

}  // namespace

TEST_CASE("Compositor factory selects the supported enum and rejects other values",
           "[core][compositor][factory]") {
    STATIC_REQUIRE(static_cast<int>(CompositorType::NONE) == 0);
    STATIC_REQUIRE(static_cast<int>(CompositorType::DEFAULT) == 1);

    SECTION("default selection constructs without initializing rendering") {
        Compositor compositor{CompositorType::DEFAULT};
        SUCCEED();
    }

    SECTION("an uninitialized default compositor may be destroyed repeatedly") {
        Compositor compositor{CompositorType::DEFAULT};
        REQUIRE(compositor.destroy() == Result::SUCCESS);
        REQUIRE(compositor.destroy() == Result::SUCCESS);
    }

    SECTION("none is not a factory selection") {
        RequireRejectedFactoryChoice(CompositorType::NONE);
    }

    SECTION("unknown enum values report the factory error") {
        RequireRejectedFactoryChoice(static_cast<CompositorType>(255));
    }

    SECTION("negative enum values report the factory error") {
        RequireRejectedFactoryChoice(static_cast<CompositorType>(-1));
    }

    // Compositor currently exposes no enum-to-string or string-to-enum API;
    // its only observable string contract is the rejected-selection diagnostic.
}

TEST_CASE("Compositor command worker starts inert and empty",
          "[core][compositor][worker][state]") {
    CompositorWorkerDouble worker;

    auto state = worker.state();
    REQUIRE_FALSE(state.running);
    REQUIRE(state.pending == 0);
    REQUIRE(state.completed == 0);

    CompositorWorkerDouble::CompletedCommand untouched;
    untouched.fn = [] {
        return Result::FATAL;
    };
    untouched.silent = true;
    untouched.result = Result::FATAL;
    untouched.message = "untouched";

    REQUIRE_FALSE(worker.take(untouched));
    REQUIRE(static_cast<bool>(untouched.fn));
    REQUIRE(untouched.silent);
    REQUIRE(untouched.result == Result::FATAL);
    REQUIRE(untouched.message == "untouched");

    worker.start();
    REQUIRE(worker.state().running);
    worker.stop();

    state = worker.state();
    REQUIRE_FALSE(state.running);
    REQUIRE(state.pending == 0);
    REQUIRE(state.completed == 0);
}

TEST_CASE("Compositor command worker supports repeated start and stop cycles",
           "[core][compositor][worker][lifecycle]") {
    CompositorWorkerDouble worker;
    std::vector<int> execution;

    REQUIRE_FALSE(worker.state().running);
    worker.stop();
    REQUIRE_FALSE(worker.state().running);

    for (int cycle = 0; cycle < 3; ++cycle) {
        worker.start();
        REQUIRE(worker.state().running);
        worker.submit([&execution, cycle] {
            execution.push_back(cycle);
            return Result::SUCCESS;
        });
        worker.stop();
        const auto state = worker.state();
        REQUIRE_FALSE(state.running);
        REQUIRE(state.pending == 0);
        REQUIRE(state.completed == static_cast<std::size_t>(cycle + 1));

        // Stopping an already stopped worker is a no-op.
        worker.stop();
        REQUIRE_FALSE(worker.state().running);
    }

    REQUIRE(execution == std::vector<int>{0, 1, 2});

    CompositorWorkerDouble::CompletedCommand command;
    for (int cycle = 0; cycle < 3; ++cycle) {
        CAPTURE(cycle);
        REQUIRE(worker.take(command));
        REQUIRE(command.result == Result::SUCCESS);
    }
    REQUIRE_FALSE(worker.take(command));
    REQUIRE(worker.state().completed == 0);
}

TEST_CASE("Compositor command worker destruction drains accepted commands",
           "[core][compositor][worker][lifecycle]") {
    std::vector<int> execution;

    {
        CompositorWorkerDouble worker;
        worker.start();
        for (int index = 0; index < 4; ++index) {
            worker.submit([&execution, index] {
                execution.push_back(index);
                return Result::SUCCESS;
            });
        }
    }

    REQUIRE(execution == std::vector<int>{0, 1, 2, 3});
}

TEST_CASE("Compositor command worker drains commands in FIFO order",
           "[core][compositor][worker][queue]") {
    CompositorWorkerDouble worker;
    std::vector<int> execution;

    worker.start();
    for (int index = 0; index < 8; ++index) {
        worker.submit([&execution, index] {
            execution.push_back(index);
            return Result::SUCCESS;
        });
    }

    // stop() must finish commands that were accepted before the stop request.
    worker.stop();
    REQUIRE(execution == std::vector<int>{0, 1, 2, 3, 4, 5, 6, 7});
    const auto state = worker.state();
    REQUIRE_FALSE(state.running);
    REQUIRE(state.pending == 0);
    REQUIRE(state.completed == 8);

    CompositorWorkerDouble::CompletedCommand command;
    for (int index = 0; index < 8; ++index) {
        CAPTURE(index);
        REQUIRE(worker.take(command));
        REQUIRE(command.result == Result::SUCCESS);
        REQUIRE_FALSE(command.fn);
    }
    REQUIRE_FALSE(worker.take(command));
}

TEST_CASE("Compositor commands may enqueue FIFO follow-up work",
           "[core][compositor][worker][queue]") {
    CompositorWorkerDouble worker;
    std::vector<int> execution;
    std::promise<void> releaseFirst;
    auto releaseFuture = releaseFirst.get_future().share();
    std::promise<void> followUpQueued;
    auto followUpQueuedFuture = followUpQueued.get_future();

    worker.start();
    worker.submit([&] {
        execution.push_back(0);
        releaseFuture.wait();
        worker.submit([&execution] {
            execution.push_back(3);
            return Result::SUCCESS;
        });
        followUpQueued.set_value();
        return Result::SUCCESS;
    });
    worker.submit([&execution] {
        execution.push_back(1);
        return Result::SUCCESS;
    });
    worker.submit([&execution] {
        execution.push_back(2);
        return Result::SUCCESS;
    });

    releaseFirst.set_value();
    const auto followUpStatus =
        followUpQueuedFuture.wait_for(std::chrono::seconds(2));
    worker.stop();

    REQUIRE(followUpStatus == std::future_status::ready);
    REQUIRE(execution == std::vector<int>{0, 1, 2, 3});
    REQUIRE(worker.state().completed == 4);

    CompositorWorkerDouble::CompletedCommand command;
    for (int index = 0; index < 4; ++index) {
        CAPTURE(index);
        REQUIRE(worker.take(command));
        REQUIRE(command.result == Result::SUCCESS);
    }
    REQUIRE_FALSE(worker.take(command));
}

TEST_CASE("Compositor command worker accepts bounded concurrent producers",
           "[core][compositor][worker][concurrency]") {
    constexpr std::size_t producerCount = 4;
    constexpr std::size_t commandsPerProducer = 12;
    constexpr std::size_t commandCount = producerCount * commandsPerProducer;

    CompositorWorkerDouble worker;
    std::atomic<std::size_t> executed{0};
    std::promise<void> releaseProducers;
    auto releaseProducersFuture = releaseProducers.get_future().share();
    std::vector<std::thread> producers;
    producers.reserve(producerCount);

    worker.start();
    for (std::size_t producer = 0; producer < producerCount; ++producer) {
        producers.emplace_back([&] {
            releaseProducersFuture.wait();
            for (std::size_t command = 0; command < commandsPerProducer; ++command) {
                worker.submit([&executed] {
                    ++executed;
                    return Result::SUCCESS;
                });
            }
        });
    }

    releaseProducers.set_value();
    for (auto& producer : producers) {
        producer.join();
    }
    worker.stop();

    REQUIRE(executed.load() == commandCount);
    const auto state = worker.state();
    REQUIRE_FALSE(state.running);
    REQUIRE(state.pending == 0);
    REQUIRE(state.completed == commandCount);

    CompositorWorkerDouble::CompletedCommand command;
    for (std::size_t index = 0; index < commandCount; ++index) {
        CAPTURE(index);
        REQUIRE(worker.take(command));
        REQUIRE(command.result == Result::SUCCESS);
        REQUIRE_FALSE(command.fn);
    }
    REQUIRE_FALSE(worker.take(command));
}

TEST_CASE("Compositor command worker never overlaps command execution",
           "[core][compositor][worker][concurrency]") {
    CompositorWorkerDouble worker;
    std::promise<void> firstStarted;
    auto firstStartedFuture = firstStarted.get_future();
    std::promise<void> releaseFirst;
    auto releaseFirstFuture = releaseFirst.get_future().share();
    std::promise<void> secondStarted;
    auto secondStartedFuture = secondStarted.get_future();

    worker.start();
    worker.submit([&] {
        firstStarted.set_value();
        releaseFirstFuture.wait();
        return Result::SUCCESS;
    });
    worker.submit([&] {
        secondStarted.set_value();
        return Result::SUCCESS;
    });

    const auto firstStartedStatus =
        firstStartedFuture.wait_for(std::chrono::seconds(2));
    auto secondStatusWhileBlocked = std::future_status::timeout;
    if (firstStartedStatus == std::future_status::ready) {
        secondStatusWhileBlocked =
            secondStartedFuture.wait_for(std::chrono::milliseconds{0});
    }
    releaseFirst.set_value();
    worker.stop();

    REQUIRE(firstStartedStatus == std::future_status::ready);
    REQUIRE(secondStatusWhileBlocked == std::future_status::timeout);
    REQUIRE(secondStartedFuture.wait_for(std::chrono::milliseconds{0}) ==
            std::future_status::ready);
    REQUIRE(worker.state().completed == 2);
}

TEST_CASE("Compositor command worker stop drains accepted work and rejects late work",
           "[core][compositor][worker][lifecycle][concurrency]") {
    CompositorWorkerDouble worker;
    std::vector<int> execution;
    bool lateCommandInvoked = false;
    std::promise<void> firstStarted;
    auto firstStartedFuture = firstStarted.get_future();
    std::promise<void> releaseFirst;
    auto releaseFirstFuture = releaseFirst.get_future().share();
    std::promise<void> stopReturned;
    auto stopReturnedFuture = stopReturned.get_future();

    worker.start();
    worker.submit([&] {
        execution.push_back(0);
        firstStarted.set_value();
        releaseFirstFuture.wait();
        return Result::SUCCESS;
    });
    worker.submit([&] {
        execution.push_back(1);
        return Result::SUCCESS;
    });
    const auto firstStartedStatus =
        firstStartedFuture.wait_for(std::chrono::seconds(2));

    std::thread stopping([&] {
        worker.stop();
        stopReturned.set_value();
    });
    const bool stoppedAccepting = worker.waitUntilStoppedAccepting();
    worker.submit([&] {
        lateCommandInvoked = true;
        return Result::SUCCESS;
    });
    const auto stopStatusWhileBlocked =
        stopReturnedFuture.wait_for(std::chrono::milliseconds{0});

    releaseFirst.set_value();
    stopping.join();

    REQUIRE(firstStartedStatus == std::future_status::ready);
    REQUIRE(stoppedAccepting);
    REQUIRE(stopStatusWhileBlocked == std::future_status::timeout);
    REQUIRE(stopReturnedFuture.wait_for(std::chrono::milliseconds{0}) ==
            std::future_status::ready);
    REQUIRE(execution == std::vector<int>{0, 1});
    REQUIRE_FALSE(lateCommandInvoked);
    const auto state = worker.state();
    REQUIRE_FALSE(state.running);
    REQUIRE(state.pending == 0);
    REQUIRE(state.completed == 2);
}

TEST_CASE("Compositor command worker ignores submissions while stopped",
           "[core][compositor][worker][queue]") {
    CompositorWorkerDouble worker;
    bool invoked = false;

    worker.submit([&invoked] {
        invoked = true;
        return Result::SUCCESS;
    });
    worker.start();
    worker.stop();
    REQUIRE_FALSE(invoked);

    worker.submit([&invoked] {
        invoked = true;
        return Result::SUCCESS;
    });
    worker.start();
    worker.stop();
    REQUIRE_FALSE(invoked);

    const auto state = worker.state();
    REQUIRE_FALSE(state.running);
    REQUIRE(state.pending == 0);
    REQUIRE(state.completed == 0);

    CompositorWorkerDouble::CompletedCommand command;
    REQUIRE_FALSE(worker.take(command));
}

TEST_CASE("Compositor command worker preserves every result and its diagnostic channel",
           "[core][compositor][worker][result]") {
    struct ExpectedCompletion {
        Result result;
        bool silent;
        const char* message;
    };
    constexpr std::array<ExpectedCompletion, 10> expected{{
        {Result::SUCCESS, false, ""},
        {Result::ERROR, false, "asynchronous error"},
        {Result::WARNING, true, "asynchronous warning"},
        {Result::FATAL, false, "asynchronous fatal result"},
        {Result::SKIP, true, ""},
        {Result::YIELD, false, ""},
        {Result::RELOAD, true, ""},
        {Result::RECREATE, false, ""},
        {Result::TIMEOUT, true, ""},
        {Result::INCOMPLETE, false, "asynchronous incomplete result"},
    }};

    JST_LOG_LAST_WARNING().clear();
    JST_LOG_LAST_ERROR().clear();
    JST_LOG_LAST_FATAL().clear();

    CompositorWorkerDouble worker;
    worker.start();
    for (const auto& entry : expected) {
        worker.submit([entry] {
            switch (entry.result) {
                case Result::WARNING:
                    JST_LOG_LAST_WARNING() = entry.message;
                    break;
                case Result::ERROR:
                case Result::INCOMPLETE:
                    JST_LOG_LAST_ERROR() = entry.message;
                    break;
                case Result::FATAL:
                    JST_LOG_LAST_FATAL() = entry.message;
                    break;
                default:
                    break;
            }
            return entry.result;
        }, entry.silent);
    }
    worker.stop();

    REQUIRE(worker.state().completed == expected.size());

    CompositorWorkerDouble::CompletedCommand command;
    for (std::size_t index = 0; index < expected.size(); ++index) {
        CAPTURE(index);
        REQUIRE(worker.take(command));
        REQUIRE(command.result == expected[index].result);
        REQUIRE(command.silent == expected[index].silent);
        REQUIRE(command.message == expected[index].message);
        REQUIRE_FALSE(command.fn);
    }
    REQUIRE_FALSE(worker.take(command));
}

TEST_CASE("Update menu opens settings and checks only from idle state",
          "[core][compositor][update][presenter]") {
    DefaultCompositorState state;
    std::vector<Mail> mail;
    DefaultCompositorCallbacks callbacks{
        .enqueueMail = [&mail](Mail&& message) {
            mail.push_back(std::move(message));
        },
    };
    PresenterContext context{state, callbacks};
    MenubarPresenter presenter{context};

    const auto dispatch = [&](bool shouldCheck) {
        mail.clear();
        auto config = presenter.build();
        config.onAction(MenubarView::Action::CheckForUpdates);

        REQUIRE(mail.size() == (shouldCheck ? 2 : 1));
        REQUIRE(std::holds_alternative<MailOpenModal>(mail[0]));
        const auto& open = std::get<MailOpenModal>(mail[0]);
        REQUIRE(open.content == ModalContent::Settings);
        REQUIRE(open.settings == SettingsSection::About);
        if (shouldCheck) {
            REQUIRE(std::holds_alternative<MailCheckForUpdates>(mail[1]));
        }
    };

    SECTION("supported idle state starts a check") {
        state.update.supported = true;
        dispatch(true);
    }

    SECTION("unsupported state only opens update settings") {
        dispatch(false);
    }

    SECTION("active and actionable states preserve the current update") {
        state.update.supported = true;

        struct Phase {
            const char* name;
            bool DefaultCompositorState::UpdateState::*flag;
        };
        const std::array phases{
            Phase{"checking", &DefaultCompositorState::UpdateState::checking},
            Phase{"available", &DefaultCompositorState::UpdateState::available},
            Phase{"downloading", &DefaultCompositorState::UpdateState::downloading},
            Phase{"ready", &DefaultCompositorState::UpdateState::ready},
            Phase{"applying", &DefaultCompositorState::UpdateState::applying},
        };

        for (const auto& phase : phases) {
            DYNAMIC_SECTION(phase.name) {
                state.update.*phase.flag = true;
                dispatch(false);
            }
        }
    }
}

TEST_CASE("Settings update actions delegate through compositor callbacks",
           "[core][compositor][update][actions]") {
    DefaultCompositorState state;
    int checks = 0;
    int downloads = 0;
    int applies = 0;
    int dismissals = 0;
    DefaultCompositorCallbacks callbacks{
        .checkForUpdates = [&checks]() {
            ++checks;
        },
        .downloadUpdate = [&downloads]() {
            ++downloads;
        },
        .applyUpdate = [&applies]() {
            ++applies;
            return false;
        },
        .dismissUpdate = [&dismissals]() {
            ++dismissals;
        },
    };
    SettingsActions actions{state, callbacks};

    REQUIRE(actions.handle(MailCheckForUpdates{}) == Result::SUCCESS);
    REQUIRE(actions.handle(MailDownloadUpdate{}) == Result::SUCCESS);
    REQUIRE(actions.handle(MailApplyUpdate{}) == Result::SUCCESS);
    REQUIRE(actions.handle(MailDismissUpdate{}) == Result::SUCCESS);

    REQUIRE(checks == 1);
    REQUIRE(downloads == 1);
    REQUIRE(applies == 1);
    REQUIRE(dismissals == 1);
}

// TODO: Inject a Compositor::Impl factory into Compositor so create argument
// forwarding, create result/exception cleanup, worker ordering, delegation of
// poll/present/destroy, and repeated initialized lifecycle behavior can be
// tested without constructing the renderer-backed default implementation.
// TODO: Inject platform dialog, file-system root, and file-availability
// operations into the default file-picker action so request mode selection,
// path confinement, stale generations, reconciliation, cancellation, and
// callback completion can be tested without native dialogs or process paths.
// TODO: Inject a flowgraph metadata reader/writer into the default stack action
// and restore path so malformed restoration, ID allocation, geometry/layout
// persistence, detached surfaces, and persistence failures are isolated from a
// concrete flowgraph and renderer configuration.
// TODO: Inject settings load/store and render-preference sinks into default
// creation and settings actions so hydration, mutation, and failure propagation
// can be tested without reading or changing the user's persisted configuration.
// TODO: Make the default action handler set injectable so Mail ownership,
// dispatch order, exact result propagation, and unhandled-mail diagnostics can
// be tested without constructing every platform- and runtime-backed action.
