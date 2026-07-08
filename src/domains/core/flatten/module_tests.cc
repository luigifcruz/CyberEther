#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "jetstream/domains/core/flatten/module.hh"
#include "jetstream/registry.hh"
#include "jetstream/testing.hh"

using namespace Jetstream;

TEST_CASE("Flatten Module - Flatten 2D to 1D F32", "[modules][flatten][F32]") {
    auto implementations = Registry::ListAvailableModules("flatten");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            TestContext ctx("flatten", impl.device, impl.runtime, impl.provider);

            Modules::Flatten config;
            ctx.setConfig(config);

            auto input = ctx.createTensor<F32>({2, 4});
            for (U64 i = 0; i < 8; ++i) {
                input.at(i / 4, i % 4) = static_cast<F32>(i);
            }

            ctx.setInput("buffer", input);

            REQUIRE(ctx.run() == Result::SUCCESS);

            auto& out = ctx.output("buffer");

            REQUIRE(out.rank() == 1);
            REQUIRE(out.shape(0) == 8);

            for (U64 i = 0; i < 8; ++i) {
                REQUIRE_THAT(out.at<F32>(i), Catch::Matchers::WithinAbs(static_cast<F32>(i), 1e-6f));
            }
        }
    }
}

TEST_CASE("Flatten Module - Flatten 3D to 1D F32", "[modules][flatten][F32][3d]") {
    auto implementations = Registry::ListAvailableModules("flatten");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            TestContext ctx("flatten", impl.device, impl.runtime, impl.provider);

            Modules::Flatten config;
            ctx.setConfig(config);

            auto input = ctx.createTensor<F32>({2, 2, 4});
            U64 idx = 0;
            for (U64 i = 0; i < 2; ++i) {
                for (U64 j = 0; j < 2; ++j) {
                    for (U64 k = 0; k < 4; ++k) {
                        input.at(i, j, k) = static_cast<F32>(idx++);
                    }
                }
            }

            ctx.setInput("buffer", input);

            REQUIRE(ctx.run() == Result::SUCCESS);

            auto& out = ctx.output("buffer");

            REQUIRE(out.rank() == 1);
            REQUIRE(out.shape(0) == 16);

            for (U64 i = 0; i < 16; ++i) {
                REQUIRE_THAT(out.at<F32>(i), Catch::Matchers::WithinAbs(static_cast<F32>(i), 1e-6f));
            }
        }
    }
}

TEST_CASE("Flatten Module - CF32", "[modules][flatten][CF32]") {
    auto implementations = Registry::ListAvailableModules("flatten");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            TestContext ctx("flatten", impl.device, impl.runtime, impl.provider);

            Modules::Flatten config;
            ctx.setConfig(config);

            auto input = ctx.createTensor<CF32>({2, 2});
            input.at(0, 0) = {0.0f, 1.0f};
            input.at(0, 1) = {2.0f, 3.0f};
            input.at(1, 0) = {4.0f, 5.0f};
            input.at(1, 1) = {6.0f, 7.0f};

            ctx.setInput("buffer", input);

            REQUIRE(ctx.run() == Result::SUCCESS);

            auto& out = ctx.output("buffer");

            REQUIRE(out.rank() == 1);
            REQUIRE(out.shape(0) == 4);

            REQUIRE_THAT(out.at<CF32>(0).real(), Catch::Matchers::WithinAbs(0.0f, 1e-6f));
            REQUIRE_THAT(out.at<CF32>(0).imag(), Catch::Matchers::WithinAbs(1.0f, 1e-6f));
            REQUIRE_THAT(out.at<CF32>(1).real(), Catch::Matchers::WithinAbs(2.0f, 1e-6f));
            REQUIRE_THAT(out.at<CF32>(1).imag(), Catch::Matchers::WithinAbs(3.0f, 1e-6f));
            REQUIRE_THAT(out.at<CF32>(2).real(), Catch::Matchers::WithinAbs(4.0f, 1e-6f));
            REQUIRE_THAT(out.at<CF32>(2).imag(), Catch::Matchers::WithinAbs(5.0f, 1e-6f));
            REQUIRE_THAT(out.at<CF32>(3).real(), Catch::Matchers::WithinAbs(6.0f, 1e-6f));
            REQUIRE_THAT(out.at<CF32>(3).imag(), Catch::Matchers::WithinAbs(7.0f, 1e-6f));
        }
    }
}
