#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "jetstream/domains/core/range/module.hh"
#include "jetstream/registry.hh"
#include "jetstream/testing.hh"

using namespace Jetstream;

TEST_CASE("Range Module - Scales Into Unit Interval", "[modules][range][F32]") {
    const auto implementations = Registry::ListAvailableModules("range");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            TestContext ctx("range", impl.device, impl.runtime, impl.provider);

            Modules::Range config;
            config.min = -2.0f;
            config.max = 2.0f;
            ctx.setConfig(config);

            auto input = ctx.createTensor<F32>({3});
            input.at(0) = -2.0f;
            input.at(1) = 0.0f;
            input.at(2) = 2.0f;

            ctx.setInput("signal", input);
            REQUIRE(ctx.run() == Result::SUCCESS);

            auto& out = ctx.output("signal");
            REQUIRE_THAT(out.at<F32>(0), Catch::Matchers::WithinAbs(0.0f, 1e-6f));
            REQUIRE_THAT(out.at<F32>(1), Catch::Matchers::WithinAbs(0.5f, 1e-6f));
            REQUIRE_THAT(out.at<F32>(2), Catch::Matchers::WithinAbs(1.0f, 1e-6f));
        }
    }
}

TEST_CASE("Range Module - Rejects Invalid Bounds", "[modules][range][error]") {
    const auto implementations = Registry::ListAvailableModules("range");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            TestContext ctx("range", impl.device, impl.runtime, impl.provider);

            Modules::Range config;
            config.min = 1.0f;
            config.max = 1.0f;
            ctx.setConfig(config);

            auto input = ctx.createTensor<F32>({2});
            ctx.setInput("signal", input);
            REQUIRE(ctx.run() == Result::ERROR);
        }
    }
}
