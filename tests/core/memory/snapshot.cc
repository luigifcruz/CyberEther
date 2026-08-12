#include <catch2/catch_test_macros.hpp>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <type_traits>
#include <utility>

#include "jetstream/tools/snapshot.hh"
#include "jetstream/types.hh"

namespace {

using namespace Jetstream;

struct PublishedRecord {
    U64 sequence = 0;
    std::string label;
};

struct SnapshotOverlapState {
    std::mutex mutex;
    std::condition_variable condition;
    bool readCopyStarted = false;
    bool publishCopyStarted = false;
    bool releaseCopies = false;
};

enum class OverlapRole {
    None,
    Read,
    Publish,
};

struct OverlappingRecord {
    U64 sequence = 0;
    std::shared_ptr<SnapshotOverlapState> state;
    OverlapRole role = OverlapRole::None;

    OverlappingRecord(U64 sequence,
                      std::shared_ptr<SnapshotOverlapState> state,
                      OverlapRole role)
        : sequence(sequence), state(std::move(state)), role(role) {}

    OverlappingRecord(const OverlappingRecord& other)
        : sequence(other.sequence), state(other.state), role(other.role) {
        if (role == OverlapRole::None) {
            return;
        }

        std::unique_lock lock(state->mutex);
        if (role == OverlapRole::Read) {
            state->readCopyStarted = true;
        } else {
            state->publishCopyStarted = true;
        }
        state->condition.notify_all();
        state->condition.wait(lock, [&]() { return state->releaseCopies; });
    }

    OverlappingRecord(OverlappingRecord&&) noexcept = default;
    OverlappingRecord& operator=(const OverlappingRecord&) = default;
    OverlappingRecord& operator=(OverlappingRecord&&) noexcept = default;
};

TEST_CASE("Snapshot publishes trivial values", "[core][memory][snapshot][trivial]") {
    STATIC_REQUIRE(!std::is_copy_constructible_v<Tools::Snapshot<U64>>);
    STATIC_REQUIRE(!std::is_move_constructible_v<Tools::Snapshot<U64>>);

    Tools::Snapshot<U64> value;
    REQUIRE(value.get() == 0);

    const U64 lvalue = 42;
    value.publish(lvalue);
    REQUIRE(value.get() == 42);

    value.publish(U64{99});
    REQUIRE(value.get() == 99);
}

TEST_CASE("Snapshot owns non-trivial values", "[core][memory][snapshot][object]") {
    STATIC_REQUIRE(!std::is_copy_assignable_v<Tools::Snapshot<std::string>>);
    STATIC_REQUIRE(!std::is_move_assignable_v<Tools::Snapshot<std::string>>);

    Tools::Snapshot<std::string> empty;
    REQUIRE(empty.get().empty());

    std::string initial = "alpha";
    Tools::Snapshot<std::string> value(initial);
    initial.assign("changed");
    REQUIRE(value.get() == "alpha");

    std::string published = "beta";
    value.publish(published);
    published.assign("changed again");
    REQUIRE(value.get() == "beta");

    auto copy = value.get();
    copy.assign("local copy");
    REQUIRE(value.get() == "beta");

    value.publish(std::string("gamma"));
    REQUIRE(value.get() == "gamma");
}

TEST_CASE("Trivial Snapshot publication crosses thread boundaries",
          "[core][memory][snapshot][threading]") {
    Tools::Snapshot<U64> value(0);

    std::thread producer([&]() {
        for (U64 sequence = 1; sequence <= 256; ++sequence) {
            value.publish(sequence);
        }
    });
    producer.join();

    REQUIRE(value.get() == 256);
}

TEST_CASE("Snapshot publication is coherent across threads",
          "[core][memory][snapshot][threading]") {
    Tools::Snapshot<PublishedRecord> value(PublishedRecord{0, "record-0"});
    std::atomic<bool> readerStarted{false};
    std::atomic<bool> publicationDone{false};
    std::atomic<bool> coherent{true};

    std::thread reader([&]() {
        readerStarted.store(true, std::memory_order_release);
        while (!publicationDone.load(std::memory_order_acquire)) {
            const auto current = value.get();
            if (current.label != "record-" + std::to_string(current.sequence)) {
                coherent.store(false, std::memory_order_relaxed);
            }
        }
    });

    const auto readerDeadline = std::chrono::steady_clock::now() +
                                std::chrono::seconds(2);
    while (!readerStarted.load(std::memory_order_acquire) &&
           std::chrono::steady_clock::now() < readerDeadline) {
        std::this_thread::yield();
    }
    const bool readerDidStart = readerStarted.load(std::memory_order_acquire);

    for (U64 sequence = 1; sequence <= 256; ++sequence) {
        value.publish(PublishedRecord{sequence, "record-" + std::to_string(sequence)});
    }
    publicationDone.store(true, std::memory_order_release);
    reader.join();

    REQUIRE(readerDidStart);
    REQUIRE(coherent.load(std::memory_order_relaxed));
    const auto final = value.get();
    REQUIRE(final.sequence == 256);
    REQUIRE(final.label == "record-256");
}

TEST_CASE("Snapshot keeps an in-flight read alive across publication",
          "[core][memory][snapshot][threading][overlap]") {
    auto state = std::make_shared<SnapshotOverlapState>();
    Tools::Snapshot<OverlappingRecord> value(
        OverlappingRecord{0, state, OverlapRole::Read});
    const OverlappingRecord next{1, state, OverlapRole::Publish};
    U64 observed = std::numeric_limits<U64>::max();

    std::thread reader([&]() {
        observed = value.get().sequence;
    });

    bool readIsInFlight = false;
    {
        std::unique_lock lock(state->mutex);
        readIsInFlight = state->condition.wait_for(
            lock, std::chrono::seconds(2), [&]() { return state->readCopyStarted; });
    }

    std::thread publisher([&]() {
        value.publish(next);
    });

    bool publishIsInFlight = false;
    {
        std::unique_lock lock(state->mutex);
        publishIsInFlight = state->condition.wait_for(
            lock, std::chrono::seconds(2), [&]() { return state->publishCopyStarted; });
    }

    {
        std::lock_guard lock(state->mutex);
        state->releaseCopies = true;
    }
    state->condition.notify_all();
    reader.join();
    publisher.join();

    REQUIRE(readIsInFlight);
    REQUIRE(publishIsInFlight);
    REQUIRE(observed == 0);
    REQUIRE(value.get().sequence == 1);
}

}  // namespace
