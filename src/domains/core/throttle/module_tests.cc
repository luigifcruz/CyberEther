#include <catch2/catch_test_macros.hpp>

#include <chrono>
#include <limits>

#include "jetstream/domains/core/throttle/module.hh"
#include "jetstream/registry.hh"
#include "jetstream/testing.hh"
#include "schedule.hh"

using namespace Jetstream;

TEST_CASE("Throttle Module - Rejects Invalid Intervals", "[modules][throttle][error]") {
    const auto implementations = Registry::ListAvailableModules("throttle");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            for (const F32 interval : {0.0f, 0.0005f, -1.0f, std::numeric_limits<F32>::quiet_NaN(),
                                       std::numeric_limits<F32>::infinity(), 172'800'000.0f}) {
                CAPTURE(interval);
                TestContext ctx("throttle", impl.device, impl.runtime, impl.provider);

                Modules::Throttle config;
                config.intervalMs = interval;
                ctx.setConfig(config);

                auto input = ctx.createTensor<F32>({4});
                ctx.setInput("buffer", input);
                REQUIRE(ctx.run() == Result::ERROR);
            }
            for (const F32 interval : {0.001f, 86'400'000.0f}) {
                CAPTURE(interval);
                TestContext ctx("throttle", impl.device, impl.runtime, impl.provider);

                Modules::Throttle config;
                config.intervalMs = interval;
                ctx.setConfig(config);

                auto input = ctx.createTensor<F32>({4});
                ctx.setInput("buffer", input);
                REQUIRE(ctx.run() == Result::SUCCESS);
            }
        }
    }
}

TEST_CASE("Throttle Module - Pass Through Across Persistent Computes",
          "[modules][throttle][timing]") {
    const auto implementations = Registry::ListAvailableModules("throttle");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            TestContext ctx("throttle", impl.device, impl.runtime, impl.provider);

            Modules::Throttle config;
            config.intervalMs = 50;
            ctx.setConfig(config);

            auto input = ctx.createTensor<F32>({8});
            for (U64 i = 0; i < 8; ++i) {
                input.at(i) = static_cast<F32>(i);
            }
            ctx.setInput("buffer", input);

            REQUIRE(ctx.start() == Result::SUCCESS);
            const auto computeStart = std::chrono::steady_clock::now();
            REQUIRE(ctx.compute() == Result::SUCCESS);
            REQUIRE(ctx.compute() == Result::SUCCESS);
            const auto computeElapsed = std::chrono::steady_clock::now() - computeStart;
            REQUIRE(computeElapsed >= std::chrono::milliseconds(25));

            auto& out = ctx.output("buffer");
            REQUIRE(out.rank() == 1);
            REQUIRE(out.shape(0) == 8);
            for (U64 i = 0; i < 8; ++i) {
                REQUIRE(out.at<F32>(i) == static_cast<F32>(i));
            }
            REQUIRE(ctx.stop() == Result::SUCCESS);
        }
    }
}

TEST_CASE("Throttle Module - Paces Sub-Millisecond Intervals On Average",
          "[modules][throttle][timing]") {
    const auto implementations = Registry::ListAvailableModules("throttle");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            TestContext ctx("throttle", impl.device, impl.runtime, impl.provider);

            Modules::Throttle config;
            config.intervalMs = 0.25f;
            ctx.setConfig(config);

            auto input = ctx.createTensor<F32>({8});
            ctx.setInput("buffer", input);

            REQUIRE(ctx.start() == Result::SUCCESS);
            const auto start = std::chrono::steady_clock::now();
            for (U64 i = 0; i < 41; ++i) {
                REQUIRE(ctx.compute() == Result::SUCCESS);
            }
            const auto elapsed = std::chrono::steady_clock::now() - start;
            REQUIRE(elapsed >= std::chrono::microseconds(9'750));
            REQUIRE(elapsed < std::chrono::seconds(1));
            REQUIRE(ctx.stop() == Result::SUCCESS);
        }
    }
}

TEST_CASE("Throttle Module - Schedules Passes By Deadline",
          "[modules][throttle][timing]") {
    using Clock = Modules::ThrottleSchedule::Clock;
    using std::chrono::milliseconds;

    // Explicit timestamps keep deadline behavior independent of host scheduling
    // and sleep overshoot on shared CI runners.
    const auto start = Clock::time_point{};
    Modules::ThrottleSchedule schedule;
    schedule.reset(milliseconds(100), start);
    REQUIRE(schedule.deadline == start);
    schedule.advance(start);
    REQUIRE(schedule.deadline == start + milliseconds(100));

    SECTION("a late pass shortens the next wait") {
        schedule.advance(start + milliseconds(150));
        REQUIRE(schedule.deadline == start + milliseconds(200));
        schedule.advance(start + milliseconds(200));
        REQUIRE(schedule.deadline == start + milliseconds(300));
    }

    SECTION("a pass exactly one interval late restarts the schedule") {
        schedule.advance(start + milliseconds(200));
        REQUIRE(schedule.deadline == start + milliseconds(300));
    }

    SECTION("a pass more than one interval late restarts the schedule") {
        schedule.advance(start + milliseconds(250));
        REQUIRE(schedule.deadline == start + milliseconds(350));
        schedule.advance(start + milliseconds(350));
        REQUIRE(schedule.deadline == start + milliseconds(450));
    }

    SECTION("sleep overshoot does not accumulate across passes") {
        schedule.advance(start + milliseconds(125));
        REQUIRE(schedule.deadline == start + milliseconds(200));
        schedule.advance(start + milliseconds(225));
        REQUIRE(schedule.deadline == start + milliseconds(300));
    }

    SECTION("a new interval lets the next pass through at once") {
        schedule.reset(milliseconds(200), start + milliseconds(50));
        REQUIRE(schedule.deadline == start + milliseconds(50));
        schedule.advance(start + milliseconds(50));
        REQUIRE(schedule.deadline == start + milliseconds(250));
    }
}

TEST_CASE("Throttle Module - Reconfigures Persistent Computes",
          "[modules][throttle][timing]") {
    TestContext ctx("throttle", DeviceType::CPU, RuntimeType::NATIVE, "generic");

    Modules::Throttle config;
    config.intervalMs = 10.0f;
    ctx.setConfig(config);

    auto input = ctx.createTensor<F32>({4});
    ctx.setInput("buffer", input);
    REQUIRE(ctx.start() == Result::SUCCESS);
    REQUIRE(ctx.compute() == Result::SUCCESS);

    const auto start = std::chrono::steady_clock::now();
    config.intervalMs = 100.0f;
    REQUIRE(ctx.reconfigure(config) == Result::SUCCESS);
    REQUIRE(ctx.compute() == Result::SUCCESS);
    REQUIRE(ctx.compute() == Result::SUCCESS);
    REQUIRE(std::chrono::steady_clock::now() - start >= std::chrono::milliseconds(100));
    REQUIRE(ctx.stop() == Result::SUCCESS);
}
