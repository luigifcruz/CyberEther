#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "jetstream/testing.hh"
#include "jetstream/registry.hh"
#include "jetstream/domains/core/arithmetic/module.hh"

using namespace Jetstream;

TEST_CASE("Arithmetic Module - Add F32", "[modules][arithmetic][F32]") {
    auto implementations = Registry::ListAvailableModules("arithmetic");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            TestContext ctx("arithmetic", impl.device, impl.runtime, impl.provider);

            Modules::Arithmetic config;
            config.operation = "add";
            config.axis = 1;
            ctx.setConfig(config);

            auto input = ctx.createTensor<F32>({2, 3});
            input.at(0, 0) = 1.0f; input.at(0, 1) = 2.0f; input.at(0, 2) = 3.0f;
            input.at(1, 0) = 4.0f; input.at(1, 1) = 5.0f; input.at(1, 2) = 6.0f;

            ctx.setInput("buffer", input);

            REQUIRE(ctx.run() == Result::SUCCESS);

            auto& out = ctx.output("buffer");
            REQUIRE(out.shape().size() == 2);
            REQUIRE(out.shape(0) == 2);
            REQUIRE(out.shape(1) == 1);

            REQUIRE_THAT(out.at<F32>(0, 0),
                         Catch::Matchers::WithinAbs(6.0f, 1e-6f));
            REQUIRE_THAT(out.at<F32>(1, 0),
                         Catch::Matchers::WithinAbs(15.0f, 1e-6f));
        }
    }
}

TEST_CASE("Arithmetic Module - Sub F32", "[modules][arithmetic][F32]") {
    auto implementations = Registry::ListAvailableModules("arithmetic");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            TestContext ctx("arithmetic", impl.device, impl.runtime, impl.provider);

            Modules::Arithmetic config;
            config.operation = "sub";
            config.axis = 0;
            ctx.setConfig(config);

            auto input = ctx.createTensor<F32>({3});
            input.at(0) = 10.0f; input.at(1) = 3.0f; input.at(2) = 2.0f;

            ctx.setInput("buffer", input);

            REQUIRE(ctx.run() == Result::SUCCESS);

            auto& out = ctx.output("buffer");
            REQUIRE(out.shape().size() == 1);
            REQUIRE(out.shape(0) == 1);

            // 0 - 10 - 3 - 2 = -15
            REQUIRE_THAT(out.at<F32>(0),
                         Catch::Matchers::WithinAbs(-15.0f, 1e-6f));
        }
    }
}

TEST_CASE("Arithmetic Module - Add F32 Squeeze", "[modules][arithmetic][F32]") {
    auto implementations = Registry::ListAvailableModules("arithmetic");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            TestContext ctx("arithmetic", impl.device, impl.runtime, impl.provider);

            Modules::Arithmetic config;
            config.operation = "add";
            config.axis = 1;
            config.squeeze = true;
            ctx.setConfig(config);

            auto input = ctx.createTensor<F32>({2, 3});
            input.at(0, 0) = 1.0f; input.at(0, 1) = 2.0f; input.at(0, 2) = 3.0f;
            input.at(1, 0) = 4.0f; input.at(1, 1) = 5.0f; input.at(1, 2) = 6.0f;

            ctx.setInput("buffer", input);

            REQUIRE(ctx.run() == Result::SUCCESS);

            auto& out = ctx.output("buffer");
            REQUIRE(out.shape().size() == 1);
            REQUIRE(out.shape(0) == 2);

            REQUIRE_THAT(out.at<F32>(0),
                         Catch::Matchers::WithinAbs(6.0f, 1e-6f));
            REQUIRE_THAT(out.at<F32>(1),
                         Catch::Matchers::WithinAbs(15.0f, 1e-6f));
        }
    }
}

TEST_CASE("Arithmetic Module - Add CF32", "[modules][arithmetic][CF32]") {
    auto implementations = Registry::ListAvailableModules("arithmetic");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            TestContext ctx("arithmetic", impl.device, impl.runtime, impl.provider);

            Modules::Arithmetic config;
            config.operation = "add";
            config.axis = 0;
            ctx.setConfig(config);

            auto input = ctx.createTensor<CF32>({3});
            input.at(0) = {1.0f, 2.0f};
            input.at(1) = {3.0f, 4.0f};
            input.at(2) = {5.0f, 6.0f};

            ctx.setInput("buffer", input);

            REQUIRE(ctx.run() == Result::SUCCESS);

            auto& out = ctx.output("buffer");
            REQUIRE(out.shape().size() == 1);
            REQUIRE(out.shape(0) == 1);

            REQUIRE_THAT(out.at<CF32>(0).real(),
                         Catch::Matchers::WithinAbs(9.0f, 1e-6f));
            REQUIRE_THAT(out.at<CF32>(0).imag(),
                         Catch::Matchers::WithinAbs(12.0f, 1e-6f));
        }
    }
}

TEST_CASE("Arithmetic Module - Invalid Operation", "[modules][arithmetic]") {
    auto implementations = Registry::ListAvailableModules("arithmetic");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            TestContext ctx("arithmetic", impl.device, impl.runtime, impl.provider);

            Modules::Arithmetic config;
            config.operation = "invalid";
            ctx.setConfig(config);

            auto input = ctx.createTensor<F32>({4});
            ctx.setInput("buffer", input);

            REQUIRE(ctx.run() == Result::ERROR);
        }
    }
}

TEST_CASE("Arithmetic Module - Invalid Axis", "[modules][arithmetic]") {
    auto implementations = Registry::ListAvailableModules("arithmetic");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            TestContext ctx("arithmetic", impl.device, impl.runtime, impl.provider);

            Modules::Arithmetic config;
            config.operation = "add";
            config.axis = 5;
            ctx.setConfig(config);

            auto input = ctx.createTensor<F32>({4});
            ctx.setInput("buffer", input);

            REQUIRE(ctx.run() == Result::ERROR);
        }
    }
}
