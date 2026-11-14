#include <catch2/catch_test_macros.hpp>

#include <array>
#include <chrono>
#include <future>
#include <thread>

#include "jetstream/platform.hh"

using namespace Jetstream;
using Clock = Platform::DeadlineWaiter::Clock;
using namespace std::chrono_literals;

TEST_CASE("Platform deadline waiter preserves notifications", "[core][platform][deadline]") {
    Platform::DeadlineWaiter waiter;
    const auto generation = waiter.generation();
    waiter.notify();
    REQUIRE(waiter.waitUntil(Clock::now() + 5s, generation));

    const auto deadline = Clock::now() + 5ms;
    REQUIRE_FALSE(waiter.waitUntil(deadline, waiter.generation()));
    REQUIRE(Clock::now() >= deadline);
    REQUIRE_FALSE(waiter.waitUntil(Clock::now() - 1s, waiter.generation()));
}

TEST_CASE("Platform deadline waiter broadcasts to concurrent waits", "[core][platform][deadline]") {
    Platform::DeadlineWaiter waiter;
    for (int iteration = 0; iteration < 10; ++iteration) {
        const auto generation = waiter.generation();
        std::array<std::future<bool>, 4> waits;
        for (auto& wait : waits) {
            wait = std::async(std::launch::async, [&] {
                return waiter.waitUntil(Clock::now() + 5s, generation);
            });
        }

        if (iteration % 2) {
            std::this_thread::sleep_for(20ms);
        }
        waiter.notify();
        for (auto& wait : waits) {
            const auto status = wait.wait_for(2s);
            const bool notified = wait.get();
            REQUIRE(status == std::future_status::ready);
            REQUIRE(notified);
        }
    }
}

TEST_CASE("Platform deadline waiters keep deadlines and notifications independent",
          "[core][platform][deadline]") {
    Platform::DeadlineWaiter first;
    Platform::DeadlineWaiter second;
    const auto generation = first.generation();
    auto shortWait = std::async(std::launch::async, [&] {
        const auto deadline = Clock::now() + 5ms;
        const bool notified = first.waitUntil(deadline, generation);
        return !notified && Clock::now() >= deadline;
    });
    auto longWait = std::async(std::launch::async, [&] {
        return first.waitUntil(Clock::now() + 5s, generation);
    });
    REQUIRE(shortWait.get());
    second.notify();
    const auto stillWaiting = longWait.wait_for(20ms);
    first.notify();
    const auto status = longWait.wait_for(2s);
    const bool notified = longWait.get();
    REQUIRE(stillWaiting == std::future_status::timeout);
    REQUIRE(status == std::future_status::ready);
    REQUIRE(notified);
}
