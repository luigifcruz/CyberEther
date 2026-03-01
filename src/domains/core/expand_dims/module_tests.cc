#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "jetstream/testing.hh"
#include "jetstream/registry.hh"
#include "jetstream/domains/core/expand_dims/module.hh"

using namespace Jetstream;

TEST_CASE("ExpandDims Module - Expand 1D to 2D at axis 0 F32", "[modules][expand_dims][F32]") {
    auto implementations = Registry::ListAvailableModules("expand_dims");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            TestContext ctx("expand_dims", impl.device, impl.runtime, impl.provider);

            Modules::ExpandDims config;
            config.axis = 0;

            ctx.setConfig(config);

            auto input = ctx.createTensor<F32>({4});
            for (U64 i = 0; i < 4; ++i) {
                input.at(i) = static_cast<F32>(i);
            }

            ctx.setInput("buffer", input);

            REQUIRE(ctx.run() == Result::SUCCESS);

            auto& out = ctx.output("buffer");

            REQUIRE(out.shape().size() == 2);
            REQUIRE(out.shape(0) == 1);
            REQUIRE(out.shape(1) == 4);

            for (U64 i = 0; i < 4; ++i) {
                REQUIRE_THAT(out.at<F32>(0, i), Catch::Matchers::WithinAbs(static_cast<F32>(i), 1e-6f));
            }
        }
    }
}

TEST_CASE("ExpandDims Module - Expand 1D to 2D at axis 1 F32", "[modules][expand_dims][F32]") {
    auto implementations = Registry::ListAvailableModules("expand_dims");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            TestContext ctx("expand_dims", impl.device, impl.runtime, impl.provider);

            Modules::ExpandDims config;
            config.axis = 1;

            ctx.setConfig(config);

            auto input = ctx.createTensor<F32>({4});
            for (U64 i = 0; i < 4; ++i) {
                input.at(i) = static_cast<F32>(i);
            }

            ctx.setInput("buffer", input);

            REQUIRE(ctx.run() == Result::SUCCESS);

            auto& out = ctx.output("buffer");

            REQUIRE(out.shape().size() == 2);
            REQUIRE(out.shape(0) == 4);
            REQUIRE(out.shape(1) == 1);

            for (U64 i = 0; i < 4; ++i) {
                REQUIRE_THAT(out.at<F32>(i, 0), Catch::Matchers::WithinAbs(static_cast<F32>(i), 1e-6f));
            }
        }
    }
}

TEST_CASE("ExpandDims Module - Expand 2D to 3D at axis 1 F32", "[modules][expand_dims][F32]") {
    auto implementations = Registry::ListAvailableModules("expand_dims");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            TestContext ctx("expand_dims", impl.device, impl.runtime, impl.provider);

            Modules::ExpandDims config;
            config.axis = 1;

            ctx.setConfig(config);

            auto input = ctx.createTensor<F32>({2, 4});
            for (U64 i = 0; i < 8; ++i) {
                input.at(i / 4, i % 4) = static_cast<F32>(i);
            }

            ctx.setInput("buffer", input);

            REQUIRE(ctx.run() == Result::SUCCESS);

            auto& out = ctx.output("buffer");

            REQUIRE(out.shape().size() == 3);
            REQUIRE(out.shape(0) == 2);
            REQUIRE(out.shape(1) == 1);
            REQUIRE(out.shape(2) == 4);

            for (U64 i = 0; i < 8; ++i) {
                REQUIRE_THAT(out.at<F32>(i / 4, 0, i % 4),
                             Catch::Matchers::WithinAbs(static_cast<F32>(i), 1e-6f));
            }
        }
    }
}

TEST_CASE("ExpandDims Module - CF32", "[modules][expand_dims][CF32]") {
    auto implementations = Registry::ListAvailableModules("expand_dims");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            TestContext ctx("expand_dims", impl.device, impl.runtime, impl.provider);

            Modules::ExpandDims config;
            config.axis = 0;

            ctx.setConfig(config);

            auto input = ctx.createTensor<CF32>({4});
            input.at(0) = {0.0f, 1.0f};
            input.at(1) = {2.0f, 3.0f};
            input.at(2) = {4.0f, 5.0f};
            input.at(3) = {6.0f, 7.0f};

            ctx.setInput("buffer", input);

            REQUIRE(ctx.run() == Result::SUCCESS);

            auto& out = ctx.output("buffer");

            REQUIRE(out.shape().size() == 2);
            REQUIRE(out.shape(0) == 1);
            REQUIRE(out.shape(1) == 4);

            REQUIRE_THAT(out.at<CF32>(0, 0).real(), Catch::Matchers::WithinAbs(0.0f, 1e-6f));
            REQUIRE_THAT(out.at<CF32>(0, 0).imag(), Catch::Matchers::WithinAbs(1.0f, 1e-6f));
            REQUIRE_THAT(out.at<CF32>(0, 1).real(), Catch::Matchers::WithinAbs(2.0f, 1e-6f));
            REQUIRE_THAT(out.at<CF32>(0, 1).imag(), Catch::Matchers::WithinAbs(3.0f, 1e-6f));
            REQUIRE_THAT(out.at<CF32>(0, 2).real(), Catch::Matchers::WithinAbs(4.0f, 1e-6f));
            REQUIRE_THAT(out.at<CF32>(0, 2).imag(), Catch::Matchers::WithinAbs(5.0f, 1e-6f));
            REQUIRE_THAT(out.at<CF32>(0, 3).real(), Catch::Matchers::WithinAbs(6.0f, 1e-6f));
            REQUIRE_THAT(out.at<CF32>(0, 3).imag(), Catch::Matchers::WithinAbs(7.0f, 1e-6f));
        }
    }
}

TEST_CASE("ExpandDims Module - Axis Out of Range Error", "[modules][expand_dims][error]") {
    auto implementations = Registry::ListAvailableModules("expand_dims");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            TestContext ctx("expand_dims", impl.device, impl.runtime, impl.provider);

            Modules::ExpandDims config;
            config.axis = 5;  // Out of range for 1D tensor

            ctx.setConfig(config);

            auto input = ctx.createTensor<F32>({4});
            ctx.setInput("buffer", input);

            REQUIRE(ctx.run() == Result::ERROR);
        }
    }
}
