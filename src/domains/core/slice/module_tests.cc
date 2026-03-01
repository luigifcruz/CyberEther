#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "jetstream/testing.hh"
#include "jetstream/registry.hh"
#include "jetstream/domains/core/slice/module.hh"

using namespace Jetstream;

TEST_CASE("Slice Module - Basic Range F32", "[modules][slice][F32]") {
    auto implementations = Registry::ListAvailableModules("slice");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            TestContext ctx("slice", impl.device, impl.runtime, impl.provider);

            Modules::Slice config;
            config.slice = "[2:6]";

            ctx.setConfig(config);

            auto input = ctx.createTensor<F32>({8});
            for (U64 i = 0; i < 8; ++i) {
                input.at(i) = static_cast<F32>(i);
            }

            ctx.setInput("buffer", input);

            REQUIRE(ctx.run() == Result::SUCCESS);

            auto& out = ctx.output("buffer");

            REQUIRE(out.shape().size() == 1);
            REQUIRE(out.shape(0) == 4);

            REQUIRE_THAT(out.at<F32>(0), Catch::Matchers::WithinAbs(2.0f, 1e-6f));
            REQUIRE_THAT(out.at<F32>(1), Catch::Matchers::WithinAbs(3.0f, 1e-6f));
            REQUIRE_THAT(out.at<F32>(2), Catch::Matchers::WithinAbs(4.0f, 1e-6f));
            REQUIRE_THAT(out.at<F32>(3), Catch::Matchers::WithinAbs(5.0f, 1e-6f));
        }
    }
}

TEST_CASE("Slice Module - Step F32", "[modules][slice][F32][step]") {
    auto implementations = Registry::ListAvailableModules("slice");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            TestContext ctx("slice", impl.device, impl.runtime, impl.provider);

            Modules::Slice config;
            config.slice = "[0:8:2]";

            ctx.setConfig(config);

            auto input = ctx.createTensor<F32>({8});
            for (U64 i = 0; i < 8; ++i) {
                input.at(i) = static_cast<F32>(i);
            }

            ctx.setInput("buffer", input);

            REQUIRE(ctx.run() == Result::SUCCESS);

            auto& out = ctx.output("buffer");

            REQUIRE(out.shape().size() == 1);
            REQUIRE(out.shape(0) == 4);

            REQUIRE_THAT(out.at<F32>(0), Catch::Matchers::WithinAbs(0.0f, 1e-6f));
            REQUIRE_THAT(out.at<F32>(1), Catch::Matchers::WithinAbs(2.0f, 1e-6f));
            REQUIRE_THAT(out.at<F32>(2), Catch::Matchers::WithinAbs(4.0f, 1e-6f));
            REQUIRE_THAT(out.at<F32>(3), Catch::Matchers::WithinAbs(6.0f, 1e-6f));
        }
    }
}

TEST_CASE("Slice Module - Single Index F32", "[modules][slice][F32][index]") {
    auto implementations = Registry::ListAvailableModules("slice");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            TestContext ctx("slice", impl.device, impl.runtime, impl.provider);

            Modules::Slice config;
            config.slice = "[1]";

            ctx.setConfig(config);

            auto input = ctx.createTensor<F32>({2, 4});
            for (U64 i = 0; i < 8; ++i) {
                input.at(i / 4, i % 4) = static_cast<F32>(i);
            }

            ctx.setInput("buffer", input);

            REQUIRE(ctx.run() == Result::SUCCESS);

            auto& out = ctx.output("buffer");

            // Selecting index 1 from a 2D tensor should reduce to 1D.
            REQUIRE(out.shape().size() == 1);
            REQUIRE(out.shape(0) == 4);
        }
    }
}

TEST_CASE("Slice Module - Ellipsis F32", "[modules][slice][F32][ellipsis]") {
    auto implementations = Registry::ListAvailableModules("slice");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            TestContext ctx("slice", impl.device, impl.runtime, impl.provider);

            Modules::Slice config;
            config.slice = "[...]";

            ctx.setConfig(config);

            auto input = ctx.createTensor<F32>({4, 4});
            for (U64 i = 0; i < 16; ++i) {
                input.at(i / 4, i % 4) = static_cast<F32>(i);
            }

            ctx.setInput("buffer", input);

            REQUIRE(ctx.run() == Result::SUCCESS);

            auto& out = ctx.output("buffer");

            REQUIRE(out.shape().size() == 2);
            REQUIRE(out.shape(0) == 4);
            REQUIRE(out.shape(1) == 4);

            for (U64 i = 0; i < 16; ++i) {
                REQUIRE_THAT(out.at<F32>(i / 4, i % 4),
                             Catch::Matchers::WithinAbs(static_cast<F32>(i), 1e-6f));
            }
        }
    }
}

TEST_CASE("Slice Module - 2D Slice F32", "[modules][slice][F32][2d]") {
    auto implementations = Registry::ListAvailableModules("slice");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            TestContext ctx("slice", impl.device, impl.runtime, impl.provider);

            Modules::Slice config;
            config.slice = "[1:3, 0:2]";

            ctx.setConfig(config);

            auto input = ctx.createTensor<F32>({4, 4});
            for (U64 i = 0; i < 16; ++i) {
                input.at(i / 4, i % 4) = static_cast<F32>(i);
            }

            ctx.setInput("buffer", input);

            REQUIRE(ctx.run() == Result::SUCCESS);

            auto& out = ctx.output("buffer");

            REQUIRE(out.shape().size() == 2);
            REQUIRE(out.shape(0) == 2);
            REQUIRE(out.shape(1) == 2);

            // Input was:
            // [[ 0,  1,  2,  3],
            //  [ 4,  5,  6,  7],
            //  [ 8,  9, 10, 11],
            //  [12, 13, 14, 15]]
            // After [1:3, 0:2]:
            // [[ 4,  5],
            //  [ 8,  9]]
            REQUIRE_THAT(out.at<F32>(0, 0), Catch::Matchers::WithinAbs(4.0f, 1e-6f));
            REQUIRE_THAT(out.at<F32>(0, 1), Catch::Matchers::WithinAbs(5.0f, 1e-6f));
            REQUIRE_THAT(out.at<F32>(1, 0), Catch::Matchers::WithinAbs(8.0f, 1e-6f));
            REQUIRE_THAT(out.at<F32>(1, 1), Catch::Matchers::WithinAbs(9.0f, 1e-6f));
        }
    }
}

TEST_CASE("Slice Module - CF32", "[modules][slice][CF32]") {
    auto implementations = Registry::ListAvailableModules("slice");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            TestContext ctx("slice", impl.device, impl.runtime, impl.provider);

            Modules::Slice config;
            config.slice = "[1:3]";

            ctx.setConfig(config);

            auto input = ctx.createTensor<CF32>({4});
            input.at(0) = {0.0f, 1.0f};
            input.at(1) = {2.0f, 3.0f};
            input.at(2) = {4.0f, 5.0f};
            input.at(3) = {6.0f, 7.0f};

            ctx.setInput("buffer", input);

            REQUIRE(ctx.run() == Result::SUCCESS);

            auto& out = ctx.output("buffer");

            REQUIRE(out.shape().size() == 1);
            REQUIRE(out.shape(0) == 2);

            REQUIRE_THAT(out.at<CF32>(0).real(), Catch::Matchers::WithinAbs(2.0f, 1e-6f));
            REQUIRE_THAT(out.at<CF32>(0).imag(), Catch::Matchers::WithinAbs(3.0f, 1e-6f));
            REQUIRE_THAT(out.at<CF32>(1).real(), Catch::Matchers::WithinAbs(4.0f, 1e-6f));
            REQUIRE_THAT(out.at<CF32>(1).imag(), Catch::Matchers::WithinAbs(5.0f, 1e-6f));
        }
    }
}

TEST_CASE("Slice Module - Validation rejects malformed slice strings",
          "[modules][slice][validation]") {
    auto implementations = Registry::ListAvailableModules("slice");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        SECTION("empty slice string") {
            TestContext ctx("slice", impl.device, impl.runtime, impl.provider);

            Modules::Slice config;
            config.slice = "";
            ctx.setConfig(config);

            auto input = ctx.createTensor<F32>({8});
            ctx.setInput("buffer", input);
            REQUIRE(ctx.run() == Result::ERROR);
        }

        SECTION("missing slice brackets") {
            TestContext ctx("slice", impl.device, impl.runtime, impl.provider);

            Modules::Slice config;
            config.slice = "1:4";
            ctx.setConfig(config);

            auto input = ctx.createTensor<F32>({8});
            ctx.setInput("buffer", input);
            REQUIRE(ctx.run() == Result::ERROR);
        }

        SECTION("invalid token") {
            TestContext ctx("slice", impl.device, impl.runtime, impl.provider);

            Modules::Slice config;
            config.slice = "[foo]";
            ctx.setConfig(config);

            auto input = ctx.createTensor<F32>({8});
            ctx.setInput("buffer", input);
            REQUIRE(ctx.run() == Result::ERROR);
        }
    }
}
