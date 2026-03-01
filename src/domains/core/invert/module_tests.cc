#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "jetstream/registry.hh"
#include "jetstream/testing.hh"

using namespace Jetstream;

TEST_CASE("Invert Module - Alternating Sign", "[modules][invert][CF32]") {
    const auto implementations = Registry::ListAvailableModules("invert");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            TestContext ctx("invert", impl.device, impl.runtime, impl.provider);

            auto input = ctx.createTensor<CF32>({5});
            input.at(0) = {1.0f, 1.0f};
            input.at(1) = {2.0f, -2.0f};
            input.at(2) = {3.0f, 3.0f};
            input.at(3) = {4.0f, -4.0f};
            input.at(4) = {5.0f, 5.0f};

            ctx.setInput("signal", input);
            REQUIRE(ctx.run() == Result::SUCCESS);

            auto& out = ctx.output("signal");
            REQUIRE_THAT(out.at<CF32>(0).real(), Catch::Matchers::WithinAbs(1.0f, 1e-6f));
            REQUIRE_THAT(out.at<CF32>(1).real(), Catch::Matchers::WithinAbs(-2.0f, 1e-6f));
            REQUIRE_THAT(out.at<CF32>(2).real(), Catch::Matchers::WithinAbs(3.0f, 1e-6f));
            REQUIRE_THAT(out.at<CF32>(3).real(), Catch::Matchers::WithinAbs(-4.0f, 1e-6f));
        }
    }
}

TEST_CASE("Invert Module - Unsupported DType Error", "[modules][invert][error]") {
    const auto implementations = Registry::ListAvailableModules("invert");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            TestContext ctx("invert", impl.device, impl.runtime, impl.provider);

            auto input = ctx.createTensor<F32>({4});
            ctx.setInput("signal", input);
            REQUIRE(ctx.run() == Result::ERROR);
        }
    }
}
