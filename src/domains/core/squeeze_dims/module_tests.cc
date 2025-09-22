#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "jetstream/testing.hh"
#include "jetstream/registry.hh"
#include "jetstream/domains/core/squeeze_dims/module.hh"

using namespace Jetstream;

TEST_CASE("SqueezeDims Module - Squeeze 2D to 1D at axis 0 F32", "[modules][squeeze_dims][F32]") {
    auto implementations = Registry::ListAvailableModules("squeeze_dims");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            TestContext ctx("squeeze_dims", impl.device, impl.runtime, impl.provider);

            Modules::SqueezeDims config;
            config.axis = 0;

            ctx.setConfig(config);

            auto input = ctx.createTensor<F32>({1, 4});
            for (U64 i = 0; i < 4; ++i) {
                input.at(0, i) = static_cast<F32>(i);
            }

            ctx.setInput("buffer", input);

            REQUIRE(ctx.run() == Result::SUCCESS);

            auto& out = ctx.output("buffer");

            REQUIRE(out.shape().size() == 1);
            REQUIRE(out.shape(0) == 4);

            for (U64 i = 0; i < 4; ++i) {
                REQUIRE_THAT(out.at<F32>(i), Catch::Matchers::WithinAbs(static_cast<F32>(i), 1e-6f));
            }
        }
    }
}

TEST_CASE("SqueezeDims Module - Squeeze 2D to 1D at axis 1 F32", "[modules][squeeze_dims][F32]") {
    auto implementations = Registry::ListAvailableModules("squeeze_dims");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            TestContext ctx("squeeze_dims", impl.device, impl.runtime, impl.provider);

            Modules::SqueezeDims config;
            config.axis = 1;

            ctx.setConfig(config);

            auto input = ctx.createTensor<F32>({4, 1});
            for (U64 i = 0; i < 4; ++i) {
                input.at(i, 0) = static_cast<F32>(i);
            }

            ctx.setInput("buffer", input);

            REQUIRE(ctx.run() == Result::SUCCESS);

            auto& out = ctx.output("buffer");

            REQUIRE(out.shape().size() == 1);
            REQUIRE(out.shape(0) == 4);

            for (U64 i = 0; i < 4; ++i) {
                REQUIRE_THAT(out.at<F32>(i), Catch::Matchers::WithinAbs(static_cast<F32>(i), 1e-6f));
            }
        }
    }
}

TEST_CASE("SqueezeDims Module - Squeeze 3D to 2D at axis 1 F32", "[modules][squeeze_dims][F32]") {
    auto implementations = Registry::ListAvailableModules("squeeze_dims");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            TestContext ctx("squeeze_dims", impl.device, impl.runtime, impl.provider);

            Modules::SqueezeDims config;
            config.axis = 1;

            ctx.setConfig(config);

            auto input = ctx.createTensor<F32>({2, 1, 4});
            for (U64 i = 0; i < 8; ++i) {
                input.at(i / 4, 0, i % 4) = static_cast<F32>(i);
            }

            ctx.setInput("buffer", input);

            REQUIRE(ctx.run() == Result::SUCCESS);

            auto& out = ctx.output("buffer");

            REQUIRE(out.shape().size() == 2);
            REQUIRE(out.shape(0) == 2);
            REQUIRE(out.shape(1) == 4);

            for (U64 i = 0; i < 8; ++i) {
                REQUIRE_THAT(out.at<F32>(i / 4, i % 4),
                             Catch::Matchers::WithinAbs(static_cast<F32>(i), 1e-6f));
            }
        }
    }
}

TEST_CASE("SqueezeDims Module - CF32", "[modules][squeeze_dims][CF32]") {
    auto implementations = Registry::ListAvailableModules("squeeze_dims");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            TestContext ctx("squeeze_dims", impl.device, impl.runtime, impl.provider);

            Modules::SqueezeDims config;
            config.axis = 0;

            ctx.setConfig(config);

            auto input = ctx.createTensor<CF32>({1, 4});
            input.at(0, 0) = {0.0f, 1.0f};
            input.at(0, 1) = {2.0f, 3.0f};
            input.at(0, 2) = {4.0f, 5.0f};
            input.at(0, 3) = {6.0f, 7.0f};

            ctx.setInput("buffer", input);

            REQUIRE(ctx.run() == Result::SUCCESS);

            auto& out = ctx.output("buffer");

            REQUIRE(out.shape().size() == 1);
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

TEST_CASE("SqueezeDims Module - Axis Out of Range Error", "[modules][squeeze_dims][error]") {
    auto implementations = Registry::ListAvailableModules("squeeze_dims");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            TestContext ctx("squeeze_dims", impl.device, impl.runtime, impl.provider);

            Modules::SqueezeDims config;
            config.axis = 5;  // Out of range for 2D tensor

            ctx.setConfig(config);

            auto input = ctx.createTensor<F32>({1, 4});
            ctx.setInput("buffer", input);

            REQUIRE(ctx.run() == Result::ERROR);
        }
    }
}

TEST_CASE("SqueezeDims Module - Dimension Not Size 1 Error", "[modules][squeeze_dims][error]") {
    auto implementations = Registry::ListAvailableModules("squeeze_dims");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            TestContext ctx("squeeze_dims", impl.device, impl.runtime, impl.provider);

            Modules::SqueezeDims config;
            config.axis = 1;  // Axis 1 has size 4, not 1

            ctx.setConfig(config);

            auto input = ctx.createTensor<F32>({1, 4});
            ctx.setInput("buffer", input);

            REQUIRE(ctx.run() == Result::ERROR);
        }
    }
}
