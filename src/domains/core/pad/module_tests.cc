#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "jetstream/testing.hh"
#include "jetstream/registry.hh"
#include "jetstream/domains/core/pad/module.hh"

using namespace Jetstream;

TEST_CASE("Pad Module - Basic 1D F32", "[modules][pad][F32]") {
    auto implementations = Registry::ListAvailableModules("pad");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            TestContext ctx("pad", impl.device, impl.runtime, impl.provider);

            Modules::Pad config;
            config.size = 4;
            config.axis = 0;

            ctx.setConfig(config);

            auto input = ctx.createTensor<F32>({4});
            input.at(0) = 1.0f;
            input.at(1) = 2.0f;
            input.at(2) = 3.0f;
            input.at(3) = 4.0f;

            ctx.setInput("unpadded", input);

            REQUIRE(ctx.run() == Result::SUCCESS);

            auto& out = ctx.output("padded");

            REQUIRE(out.shape().size() == 1);
            REQUIRE(out.shape(0) == 8);

            // Original data.
            REQUIRE_THAT(out.at<F32>(0), Catch::Matchers::WithinAbs(1.0f, 1e-6f));
            REQUIRE_THAT(out.at<F32>(1), Catch::Matchers::WithinAbs(2.0f, 1e-6f));
            REQUIRE_THAT(out.at<F32>(2), Catch::Matchers::WithinAbs(3.0f, 1e-6f));
            REQUIRE_THAT(out.at<F32>(3), Catch::Matchers::WithinAbs(4.0f, 1e-6f));

            // Padding (zeros).
            REQUIRE_THAT(out.at<F32>(4), Catch::Matchers::WithinAbs(0.0f, 1e-6f));
            REQUIRE_THAT(out.at<F32>(5), Catch::Matchers::WithinAbs(0.0f, 1e-6f));
            REQUIRE_THAT(out.at<F32>(6), Catch::Matchers::WithinAbs(0.0f, 1e-6f));
            REQUIRE_THAT(out.at<F32>(7), Catch::Matchers::WithinAbs(0.0f, 1e-6f));
        }
    }
}

TEST_CASE("Pad Module - Zero Padding F32", "[modules][pad][F32]") {
    auto implementations = Registry::ListAvailableModules("pad");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            TestContext ctx("pad", impl.device, impl.runtime, impl.provider);

            Modules::Pad config;
            config.size = 0;
            config.axis = 0;

            ctx.setConfig(config);

            auto input = ctx.createTensor<F32>({4});
            input.at(0) = 1.0f;
            input.at(1) = 2.0f;
            input.at(2) = 3.0f;
            input.at(3) = 4.0f;

            ctx.setInput("unpadded", input);

            REQUIRE(ctx.run() == Result::SUCCESS);

            auto& out = ctx.output("padded");

            REQUIRE(out.shape().size() == 1);
            REQUIRE(out.shape(0) == 4);

            for (U64 i = 0; i < 4; ++i) {
                REQUIRE_THAT(out.at<F32>(i),
                             Catch::Matchers::WithinAbs(static_cast<F32>(i + 1), 1e-6f));
            }
        }
    }
}

TEST_CASE("Pad Module - 2D Axis 0 F32", "[modules][pad][F32][2d]") {
    auto implementations = Registry::ListAvailableModules("pad");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            TestContext ctx("pad", impl.device, impl.runtime, impl.provider);

            Modules::Pad config;
            config.size = 2;
            config.axis = 0;

            ctx.setConfig(config);

            auto input = ctx.createTensor<F32>({2, 3});
            // Row 0: 1, 2, 3
            // Row 1: 4, 5, 6
            for (U64 i = 0; i < 6; ++i) {
                input.at(i / 3, i % 3) = static_cast<F32>(i + 1);
            }

            ctx.setInput("unpadded", input);

            REQUIRE(ctx.run() == Result::SUCCESS);

            auto& out = ctx.output("padded");

            REQUIRE(out.shape().size() == 2);
            REQUIRE(out.shape(0) == 4);
            REQUIRE(out.shape(1) == 3);

            // Original rows.
            REQUIRE_THAT(out.at<F32>(0, 0), Catch::Matchers::WithinAbs(1.0f, 1e-6f));
            REQUIRE_THAT(out.at<F32>(0, 1), Catch::Matchers::WithinAbs(2.0f, 1e-6f));
            REQUIRE_THAT(out.at<F32>(0, 2), Catch::Matchers::WithinAbs(3.0f, 1e-6f));
            REQUIRE_THAT(out.at<F32>(1, 0), Catch::Matchers::WithinAbs(4.0f, 1e-6f));
            REQUIRE_THAT(out.at<F32>(1, 1), Catch::Matchers::WithinAbs(5.0f, 1e-6f));
            REQUIRE_THAT(out.at<F32>(1, 2), Catch::Matchers::WithinAbs(6.0f, 1e-6f));

            // Padded rows (zeros).
            for (U64 col = 0; col < 3; ++col) {
                REQUIRE_THAT(out.at<F32>(2, col), Catch::Matchers::WithinAbs(0.0f, 1e-6f));
                REQUIRE_THAT(out.at<F32>(3, col), Catch::Matchers::WithinAbs(0.0f, 1e-6f));
            }
        }
    }
}

TEST_CASE("Pad Module - 2D Axis 1 F32", "[modules][pad][F32][2d]") {
    auto implementations = Registry::ListAvailableModules("pad");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            TestContext ctx("pad", impl.device, impl.runtime, impl.provider);

            Modules::Pad config;
            config.size = 2;
            config.axis = 1;

            ctx.setConfig(config);

            auto input = ctx.createTensor<F32>({2, 3});
            // Row 0: 1, 2, 3
            // Row 1: 4, 5, 6
            for (U64 i = 0; i < 6; ++i) {
                input.at(i / 3, i % 3) = static_cast<F32>(i + 1);
            }

            ctx.setInput("unpadded", input);

            REQUIRE(ctx.run() == Result::SUCCESS);

            auto& out = ctx.output("padded");

            REQUIRE(out.shape().size() == 2);
            REQUIRE(out.shape(0) == 2);
            REQUIRE(out.shape(1) == 5);

            // Row 0: 1, 2, 3, 0, 0
            REQUIRE_THAT(out.at<F32>(0, 0), Catch::Matchers::WithinAbs(1.0f, 1e-6f));
            REQUIRE_THAT(out.at<F32>(0, 1), Catch::Matchers::WithinAbs(2.0f, 1e-6f));
            REQUIRE_THAT(out.at<F32>(0, 2), Catch::Matchers::WithinAbs(3.0f, 1e-6f));
            REQUIRE_THAT(out.at<F32>(0, 3), Catch::Matchers::WithinAbs(0.0f, 1e-6f));
            REQUIRE_THAT(out.at<F32>(0, 4), Catch::Matchers::WithinAbs(0.0f, 1e-6f));

            // Row 1: 4, 5, 6, 0, 0
            REQUIRE_THAT(out.at<F32>(1, 0), Catch::Matchers::WithinAbs(4.0f, 1e-6f));
            REQUIRE_THAT(out.at<F32>(1, 1), Catch::Matchers::WithinAbs(5.0f, 1e-6f));
            REQUIRE_THAT(out.at<F32>(1, 2), Catch::Matchers::WithinAbs(6.0f, 1e-6f));
            REQUIRE_THAT(out.at<F32>(1, 3), Catch::Matchers::WithinAbs(0.0f, 1e-6f));
            REQUIRE_THAT(out.at<F32>(1, 4), Catch::Matchers::WithinAbs(0.0f, 1e-6f));
        }
    }
}

TEST_CASE("Pad Module - CF32", "[modules][pad][CF32]") {
    auto implementations = Registry::ListAvailableModules("pad");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            TestContext ctx("pad", impl.device, impl.runtime, impl.provider);

            Modules::Pad config;
            config.size = 2;
            config.axis = 0;

            ctx.setConfig(config);

            auto input = ctx.createTensor<CF32>({2});
            input.at(0) = {1.0f, 2.0f};
            input.at(1) = {3.0f, 4.0f};

            ctx.setInput("unpadded", input);

            REQUIRE(ctx.run() == Result::SUCCESS);

            auto& out = ctx.output("padded");

            REQUIRE(out.shape().size() == 1);
            REQUIRE(out.shape(0) == 4);

            // Original data.
            REQUIRE_THAT(out.at<CF32>(0).real(), Catch::Matchers::WithinAbs(1.0f, 1e-6f));
            REQUIRE_THAT(out.at<CF32>(0).imag(), Catch::Matchers::WithinAbs(2.0f, 1e-6f));
            REQUIRE_THAT(out.at<CF32>(1).real(), Catch::Matchers::WithinAbs(3.0f, 1e-6f));
            REQUIRE_THAT(out.at<CF32>(1).imag(), Catch::Matchers::WithinAbs(4.0f, 1e-6f));

            // Padding (zeros).
            REQUIRE_THAT(out.at<CF32>(2).real(), Catch::Matchers::WithinAbs(0.0f, 1e-6f));
            REQUIRE_THAT(out.at<CF32>(2).imag(), Catch::Matchers::WithinAbs(0.0f, 1e-6f));
            REQUIRE_THAT(out.at<CF32>(3).real(), Catch::Matchers::WithinAbs(0.0f, 1e-6f));
            REQUIRE_THAT(out.at<CF32>(3).imag(), Catch::Matchers::WithinAbs(0.0f, 1e-6f));
        }
    }
}

TEST_CASE("Pad Module - Validation rejects invalid axis",
          "[modules][pad][validation]") {
    auto implementations = Registry::ListAvailableModules("pad");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: "
                        << impl.runtime) {
            TestContext ctx("pad", impl.device, impl.runtime, impl.provider);

            Modules::Pad config;
            config.size = 2;
            config.axis = 2;
            ctx.setConfig(config);

            auto input = ctx.createTensor<F32>({2, 3});
            ctx.setInput("unpadded", input);

            REQUIRE(ctx.run() == Result::ERROR);
        }
    }
}
