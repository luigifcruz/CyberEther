#include <catch2/catch_test_macros.hpp>

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

TEST_CASE("Throttle Module - Pass Through Across Multiple Runs",
          "[modules][throttle][timing]") {
    const auto implementations = Registry::ListAvailableModules("throttle");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            TestContext ctx("throttle", impl.device, impl.runtime, impl.provider);

            Modules::Throttle config;
            config.intervalMs = 10;
            ctx.setConfig(config);

            auto input = ctx.createTensor<F32>({8});
            for (U64 i = 0; i < 8; ++i) {
                input.at(i) = static_cast<F32>(i);
            }
            ctx.setInput("buffer", input);

            REQUIRE(ctx.run() == Result::SUCCESS);

            REQUIRE(ctx.run() == Result::SUCCESS);
            auto& out = ctx.output("buffer");
            REQUIRE(out.rank() == 1);
            REQUIRE(out.shape(0) == 8);
        }
    }
}
