#include <catch2/catch_test_macros.hpp>

#include <chrono>

#include "jetstream/domains/core/throttle/module.hh"
#include "jetstream/registry.hh"
#include "jetstream/testing.hh"

using namespace Jetstream;

TEST_CASE("Throttle Module - Rejects Zero Interval", "[modules][throttle][error]") {
    const auto implementations = Registry::ListAvailableModules("throttle");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            TestContext ctx("throttle", impl.device, impl.runtime, impl.provider);

            Modules::Throttle config;
            config.intervalMs = 0;
            ctx.setConfig(config);

            auto input = ctx.createTensor<F32>({4});
            ctx.setInput("buffer", input);
            REQUIRE(ctx.run() == Result::ERROR);
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
