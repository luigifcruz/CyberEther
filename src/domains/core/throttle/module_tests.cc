#include <catch2/catch_test_macros.hpp>

#include <chrono>
#include <limits>
#include <thread>

#include "jetstream/domains/core/throttle/module.hh"
#include "jetstream/registry.hh"
#include "jetstream/testing.hh"

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
    using Clock = std::chrono::steady_clock;
    using std::chrono::milliseconds;

    TestContext ctx("throttle", DeviceType::CPU, RuntimeType::NATIVE, "generic");

    Modules::Throttle config;
    config.intervalMs = 100.0f;
    ctx.setConfig(config);

    auto input = ctx.createTensor<F32>({4});
    ctx.setInput("buffer", input);
    REQUIRE(ctx.start() == Result::SUCCESS);

    const auto timed = [&] {
        const auto start = Clock::now();
        REQUIRE(ctx.compute() == Result::SUCCESS);
        return Clock::now() - start;
    };

    SECTION("a late pass shortens the next wait") {
        REQUIRE(timed() < milliseconds(50));
        std::this_thread::sleep_for(milliseconds(150));
        REQUIRE(timed() < milliseconds(50));
        const auto wait = timed();
        REQUIRE(wait >= milliseconds(30));
        REQUIRE(wait < milliseconds(90));
    }

    SECTION("a pass more than one interval late restarts the schedule") {
        REQUIRE(timed() < milliseconds(50));
        std::this_thread::sleep_for(milliseconds(250));
        REQUIRE(timed() < milliseconds(50));
        REQUIRE(timed() >= milliseconds(90));
    }

    SECTION("a new interval lets the next pass through at once") {
        REQUIRE(timed() < milliseconds(50));
        config.intervalMs = 200.0f;
        REQUIRE(ctx.reconfigure(config) == Result::SUCCESS);
        REQUIRE(timed() < milliseconds(50));
        REQUIRE(timed() >= milliseconds(190));
    }

    REQUIRE(ctx.stop() == Result::SUCCESS);
}
