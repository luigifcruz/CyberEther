#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "jetstream/domains/core/duplicate/module.hh"
#include "jetstream/registry.hh"
#include "jetstream/testing.hh"

using namespace Jetstream;

TEST_CASE("Duplicate Module - F32", "[modules][duplicate][F32]") {
    const auto implementations = Registry::ListAvailableModules("duplicate");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            TestContext ctx("duplicate", impl.device, impl.runtime, impl.provider);

            Modules::Duplicate config;
            config.hostAccessible = true;
            ctx.setConfig(config);

            auto input = ctx.createTensor<F32>({4});
            input.at(0) = 1.0f;
            input.at(1) = -2.0f;
            input.at(2) = 3.0f;
            input.at(3) = -4.0f;

            ctx.setInput("buffer", input);
            REQUIRE(ctx.run() == Result::SUCCESS);

            auto& out = ctx.output("buffer");
            REQUIRE(out.shape(0) == 4);
            REQUIRE_THAT(out.at<F32>(0), Catch::Matchers::WithinAbs(1.0f, 1e-6f));
            REQUIRE_THAT(out.at<F32>(3), Catch::Matchers::WithinAbs(-4.0f, 1e-6f));
        }
    }
}

TEST_CASE("Duplicate Module - Unsupported Data Type Error",
          "[modules][duplicate][error]") {
    const auto implementations = Registry::ListAvailableModules("duplicate");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            TestContext ctx("duplicate", impl.device, impl.runtime, impl.provider);

            auto input = ctx.createTensor<CI8>({2});
            input.at(0) = {1, 2};
            input.at(1) = {3, 4};

            ctx.setInput("buffer", input);
            REQUIRE(ctx.run() == Result::ERROR);
        }
    }
}
