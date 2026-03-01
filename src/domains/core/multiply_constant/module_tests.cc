#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "jetstream/testing.hh"
#include "jetstream/registry.hh"
#include "jetstream/domains/core/multiply_constant/module.hh"

using namespace Jetstream;

TEST_CASE("MultiplyConstant Module - Basic F32", "[modules][multiply_constant][F32]") {
    auto implementations = Registry::ListAvailableModules("multiply_constant");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            TestContext ctx("multiply_constant", impl.device, impl.runtime, impl.provider);

            Modules::MultiplyConstant config;
            config.constant = 2.0f;

            ctx.setConfig(config);

            auto input = ctx.createTensor<F32>({4});
            input.at(0) = 1.0f;
            input.at(1) = 2.0f;
            input.at(2) = 3.0f;
            input.at(3) = 4.0f;

            ctx.setInput("factor", input);

            REQUIRE(ctx.run() == Result::SUCCESS);

            auto& out = ctx.output("product");

            REQUIRE(out.shape().size() == 1);
            REQUIRE(out.shape(0) == 4);

            REQUIRE_THAT(out.at<F32>(0), Catch::Matchers::WithinAbs(2.0f, 1e-6f));
            REQUIRE_THAT(out.at<F32>(1), Catch::Matchers::WithinAbs(4.0f, 1e-6f));
            REQUIRE_THAT(out.at<F32>(2), Catch::Matchers::WithinAbs(6.0f, 1e-6f));
            REQUIRE_THAT(out.at<F32>(3), Catch::Matchers::WithinAbs(8.0f, 1e-6f));
        }
    }
}

TEST_CASE("MultiplyConstant Module - Attenuation F32", "[modules][multiply_constant][F32]") {
    auto implementations = Registry::ListAvailableModules("multiply_constant");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            TestContext ctx("multiply_constant", impl.device, impl.runtime, impl.provider);

            Modules::MultiplyConstant config;
            config.constant = 0.5f;

            ctx.setConfig(config);

            auto input = ctx.createTensor<F32>({4});
            input.at(0) = 2.0f;
            input.at(1) = 4.0f;
            input.at(2) = 6.0f;
            input.at(3) = 8.0f;

            ctx.setInput("factor", input);

            REQUIRE(ctx.run() == Result::SUCCESS);

            auto& out = ctx.output("product");

            REQUIRE_THAT(out.at<F32>(0), Catch::Matchers::WithinAbs(1.0f, 1e-6f));
            REQUIRE_THAT(out.at<F32>(1), Catch::Matchers::WithinAbs(2.0f, 1e-6f));
            REQUIRE_THAT(out.at<F32>(2), Catch::Matchers::WithinAbs(3.0f, 1e-6f));
            REQUIRE_THAT(out.at<F32>(3), Catch::Matchers::WithinAbs(4.0f, 1e-6f));
        }
    }
}

TEST_CASE("MultiplyConstant Module - Negative Constant F32", "[modules][multiply_constant][F32]") {
    auto implementations = Registry::ListAvailableModules("multiply_constant");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            TestContext ctx("multiply_constant", impl.device, impl.runtime, impl.provider);

            Modules::MultiplyConstant config;
            config.constant = -1.0f;

            ctx.setConfig(config);

            auto input = ctx.createTensor<F32>({3});
            input.at(0) = 1.0f;
            input.at(1) = -2.0f;
            input.at(2) = 3.0f;

            ctx.setInput("factor", input);

            REQUIRE(ctx.run() == Result::SUCCESS);

            auto& out = ctx.output("product");

            REQUIRE_THAT(out.at<F32>(0), Catch::Matchers::WithinAbs(-1.0f, 1e-6f));
            REQUIRE_THAT(out.at<F32>(1), Catch::Matchers::WithinAbs(2.0f, 1e-6f));
            REQUIRE_THAT(out.at<F32>(2), Catch::Matchers::WithinAbs(-3.0f, 1e-6f));
        }
    }
}

TEST_CASE("MultiplyConstant Module - CF32", "[modules][multiply_constant][CF32]") {
    auto implementations = Registry::ListAvailableModules("multiply_constant");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            TestContext ctx("multiply_constant", impl.device, impl.runtime, impl.provider);

            Modules::MultiplyConstant config;
            config.constant = 2.0f;

            ctx.setConfig(config);

            auto input = ctx.createTensor<CF32>({2});
            input.at(0) = {1.0f, 2.0f};
            input.at(1) = {3.0f, 4.0f};

            ctx.setInput("factor", input);

            REQUIRE(ctx.run() == Result::SUCCESS);

            auto& out = ctx.output("product");

            REQUIRE(out.shape().size() == 1);
            REQUIRE(out.shape(0) == 2);

            REQUIRE_THAT(out.at<CF32>(0).real(), Catch::Matchers::WithinAbs(2.0f, 1e-6f));
            REQUIRE_THAT(out.at<CF32>(0).imag(), Catch::Matchers::WithinAbs(4.0f, 1e-6f));
            REQUIRE_THAT(out.at<CF32>(1).real(), Catch::Matchers::WithinAbs(6.0f, 1e-6f));
            REQUIRE_THAT(out.at<CF32>(1).imag(), Catch::Matchers::WithinAbs(8.0f, 1e-6f));
        }
    }
}

TEST_CASE("MultiplyConstant Module - 2D Tensor F32", "[modules][multiply_constant][F32][2d]") {
    auto implementations = Registry::ListAvailableModules("multiply_constant");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            TestContext ctx("multiply_constant", impl.device, impl.runtime, impl.provider);

            Modules::MultiplyConstant config;
            config.constant = 3.0f;

            ctx.setConfig(config);

            auto input = ctx.createTensor<F32>({2, 3});
            for (U64 i = 0; i < 6; ++i) {
                input.at(i / 3, i % 3) = static_cast<F32>(i + 1);
            }

            ctx.setInput("factor", input);

            REQUIRE(ctx.run() == Result::SUCCESS);

            auto& out = ctx.output("product");

            REQUIRE(out.shape().size() == 2);
            REQUIRE(out.shape(0) == 2);
            REQUIRE(out.shape(1) == 3);

            for (U64 i = 0; i < 6; ++i) {
                REQUIRE_THAT(out.at<F32>(i / 3, i % 3),
                             Catch::Matchers::WithinAbs(static_cast<F32>((i + 1) * 3), 1e-6f));
            }
        }
    }
}

TEST_CASE("MultiplyConstant Module - Zero Constant F32", "[modules][multiply_constant][F32]") {
    auto implementations = Registry::ListAvailableModules("multiply_constant");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            TestContext ctx("multiply_constant", impl.device, impl.runtime, impl.provider);

            Modules::MultiplyConstant config;
            config.constant = 0.0f;

            ctx.setConfig(config);

            auto input = ctx.createTensor<F32>({4});
            input.at(0) = 1.0f;
            input.at(1) = 2.0f;
            input.at(2) = 3.0f;
            input.at(3) = 4.0f;

            ctx.setInput("factor", input);

            REQUIRE(ctx.run() == Result::SUCCESS);

            auto& out = ctx.output("product");

            for (U64 i = 0; i < 4; ++i) {
                REQUIRE_THAT(out.at<F32>(i), Catch::Matchers::WithinAbs(0.0f, 1e-6f));
            }
        }
    }
}
