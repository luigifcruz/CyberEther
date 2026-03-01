#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "jetstream/testing.hh"
#include "jetstream/registry.hh"
#include "jetstream/domains/core/unpad/module.hh"

using namespace Jetstream;

TEST_CASE("Unpad Module - Basic 1D F32", "[modules][unpad][F32]") {
    auto implementations = Registry::ListAvailableModules("unpad");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            TestContext ctx("unpad", impl.device, impl.runtime, impl.provider);

            Modules::Unpad config;
            config.size = 4;
            config.axis = 0;

            ctx.setConfig(config);

            auto input = ctx.createTensor<F32>({8});
            // Data: 1, 2, 3, 4, 5, 6, 7, 8
            for (U64 i = 0; i < 8; ++i) {
                input.at(i) = static_cast<F32>(i + 1);
            }

            ctx.setInput("padded", input);

            REQUIRE(ctx.run() == Result::SUCCESS);

            auto& outUnpadded = ctx.output("unpadded");
            auto& outPad = ctx.output("pad");

            // Unpadded output should be first 4 elements.
            REQUIRE(outUnpadded.shape().size() == 1);
            REQUIRE(outUnpadded.shape(0) == 4);

            REQUIRE_THAT(outUnpadded.at<F32>(0), Catch::Matchers::WithinAbs(1.0f, 1e-6f));
            REQUIRE_THAT(outUnpadded.at<F32>(1), Catch::Matchers::WithinAbs(2.0f, 1e-6f));
            REQUIRE_THAT(outUnpadded.at<F32>(2), Catch::Matchers::WithinAbs(3.0f, 1e-6f));
            REQUIRE_THAT(outUnpadded.at<F32>(3), Catch::Matchers::WithinAbs(4.0f, 1e-6f));

            // Pad output should be last 4 elements.
            REQUIRE(outPad.shape().size() == 1);
            REQUIRE(outPad.shape(0) == 4);

            REQUIRE_THAT(outPad.at<F32>(0), Catch::Matchers::WithinAbs(5.0f, 1e-6f));
            REQUIRE_THAT(outPad.at<F32>(1), Catch::Matchers::WithinAbs(6.0f, 1e-6f));
            REQUIRE_THAT(outPad.at<F32>(2), Catch::Matchers::WithinAbs(7.0f, 1e-6f));
            REQUIRE_THAT(outPad.at<F32>(3), Catch::Matchers::WithinAbs(8.0f, 1e-6f));
        }
    }
}

TEST_CASE("Unpad Module - Zero Size F32", "[modules][unpad][F32]") {
    auto implementations = Registry::ListAvailableModules("unpad");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            TestContext ctx("unpad", impl.device, impl.runtime, impl.provider);

            Modules::Unpad config;
            config.size = 0;
            config.axis = 0;

            ctx.setConfig(config);

            auto input = ctx.createTensor<F32>({4});
            for (U64 i = 0; i < 4; ++i) {
                input.at(i) = static_cast<F32>(i + 1);
            }

            ctx.setInput("padded", input);

            REQUIRE(ctx.run() == Result::SUCCESS);

            auto& outUnpadded = ctx.output("unpadded");
            auto& outPad = ctx.output("pad");

            REQUIRE(outUnpadded.shape().size() == 1);
            REQUIRE(outUnpadded.shape(0) == 4);
            REQUIRE(outPad.shape().size() == 1);
            REQUIRE(outPad.shape(0) == 0);

            for (U64 i = 0; i < 4; ++i) {
                REQUIRE_THAT(outUnpadded.at<F32>(i),
                             Catch::Matchers::WithinAbs(static_cast<F32>(i + 1), 1e-6f));
            }
        }
    }
}

TEST_CASE("Unpad Module - 2D Axis 0 F32", "[modules][unpad][F32][2d]") {
    auto implementations = Registry::ListAvailableModules("unpad");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            TestContext ctx("unpad", impl.device, impl.runtime, impl.provider);

            Modules::Unpad config;
            config.size = 2;
            config.axis = 0;

            ctx.setConfig(config);

            auto input = ctx.createTensor<F32>({4, 3});
            // Rows 0-3, 3 columns each.
            for (U64 i = 0; i < 12; ++i) {
                input.at(i / 3, i % 3) = static_cast<F32>(i + 1);
            }

            ctx.setInput("padded", input);

            REQUIRE(ctx.run() == Result::SUCCESS);

            auto& outUnpadded = ctx.output("unpadded");
            auto& outPad = ctx.output("pad");

            // Unpadded: first 2 rows.
            REQUIRE(outUnpadded.shape().size() == 2);
            REQUIRE(outUnpadded.shape(0) == 2);
            REQUIRE(outUnpadded.shape(1) == 3);

            REQUIRE_THAT(outUnpadded.at<F32>(0, 0), Catch::Matchers::WithinAbs(1.0f, 1e-6f));
            REQUIRE_THAT(outUnpadded.at<F32>(0, 1), Catch::Matchers::WithinAbs(2.0f, 1e-6f));
            REQUIRE_THAT(outUnpadded.at<F32>(0, 2), Catch::Matchers::WithinAbs(3.0f, 1e-6f));
            REQUIRE_THAT(outUnpadded.at<F32>(1, 0), Catch::Matchers::WithinAbs(4.0f, 1e-6f));
            REQUIRE_THAT(outUnpadded.at<F32>(1, 1), Catch::Matchers::WithinAbs(5.0f, 1e-6f));
            REQUIRE_THAT(outUnpadded.at<F32>(1, 2), Catch::Matchers::WithinAbs(6.0f, 1e-6f));

            // Pad: last 2 rows.
            REQUIRE(outPad.shape().size() == 2);
            REQUIRE(outPad.shape(0) == 2);
            REQUIRE(outPad.shape(1) == 3);

            REQUIRE_THAT(outPad.at<F32>(0, 0), Catch::Matchers::WithinAbs(7.0f, 1e-6f));
            REQUIRE_THAT(outPad.at<F32>(0, 1), Catch::Matchers::WithinAbs(8.0f, 1e-6f));
            REQUIRE_THAT(outPad.at<F32>(0, 2), Catch::Matchers::WithinAbs(9.0f, 1e-6f));
            REQUIRE_THAT(outPad.at<F32>(1, 0), Catch::Matchers::WithinAbs(10.0f, 1e-6f));
            REQUIRE_THAT(outPad.at<F32>(1, 1), Catch::Matchers::WithinAbs(11.0f, 1e-6f));
            REQUIRE_THAT(outPad.at<F32>(1, 2), Catch::Matchers::WithinAbs(12.0f, 1e-6f));
        }
    }
}

TEST_CASE("Unpad Module - 2D Axis 1 F32", "[modules][unpad][F32][2d]") {
    auto implementations = Registry::ListAvailableModules("unpad");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            TestContext ctx("unpad", impl.device, impl.runtime, impl.provider);

            Modules::Unpad config;
            config.size = 2;
            config.axis = 1;

            ctx.setConfig(config);

            auto input = ctx.createTensor<F32>({2, 5});
            // Row 0: 1, 2, 3, 4, 5
            // Row 1: 6, 7, 8, 9, 10
            for (U64 i = 0; i < 10; ++i) {
                input.at(i / 5, i % 5) = static_cast<F32>(i + 1);
            }

            ctx.setInput("padded", input);

            REQUIRE(ctx.run() == Result::SUCCESS);

            auto& outUnpadded = ctx.output("unpadded");
            auto& outPad = ctx.output("pad");

            // Unpadded: 2 rows, 3 columns.
            REQUIRE(outUnpadded.shape().size() == 2);
            REQUIRE(outUnpadded.shape(0) == 2);
            REQUIRE(outUnpadded.shape(1) == 3);

            // Row 0: 1, 2, 3
            REQUIRE_THAT(outUnpadded.at<F32>(0, 0), Catch::Matchers::WithinAbs(1.0f, 1e-6f));
            REQUIRE_THAT(outUnpadded.at<F32>(0, 1), Catch::Matchers::WithinAbs(2.0f, 1e-6f));
            REQUIRE_THAT(outUnpadded.at<F32>(0, 2), Catch::Matchers::WithinAbs(3.0f, 1e-6f));

            // Row 1: 6, 7, 8
            REQUIRE_THAT(outUnpadded.at<F32>(1, 0), Catch::Matchers::WithinAbs(6.0f, 1e-6f));
            REQUIRE_THAT(outUnpadded.at<F32>(1, 1), Catch::Matchers::WithinAbs(7.0f, 1e-6f));
            REQUIRE_THAT(outUnpadded.at<F32>(1, 2), Catch::Matchers::WithinAbs(8.0f, 1e-6f));

            // Pad: 2 rows, 2 columns.
            REQUIRE(outPad.shape().size() == 2);
            REQUIRE(outPad.shape(0) == 2);
            REQUIRE(outPad.shape(1) == 2);

            // Row 0: 4, 5
            REQUIRE_THAT(outPad.at<F32>(0, 0), Catch::Matchers::WithinAbs(4.0f, 1e-6f));
            REQUIRE_THAT(outPad.at<F32>(0, 1), Catch::Matchers::WithinAbs(5.0f, 1e-6f));

            // Row 1: 9, 10
            REQUIRE_THAT(outPad.at<F32>(1, 0), Catch::Matchers::WithinAbs(9.0f, 1e-6f));
            REQUIRE_THAT(outPad.at<F32>(1, 1), Catch::Matchers::WithinAbs(10.0f, 1e-6f));
        }
    }
}

TEST_CASE("Unpad Module - CF32", "[modules][unpad][CF32]") {
    auto implementations = Registry::ListAvailableModules("unpad");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            TestContext ctx("unpad", impl.device, impl.runtime, impl.provider);

            Modules::Unpad config;
            config.size = 2;
            config.axis = 0;

            ctx.setConfig(config);

            auto input = ctx.createTensor<CF32>({4});
            input.at(0) = {1.0f, 2.0f};
            input.at(1) = {3.0f, 4.0f};
            input.at(2) = {5.0f, 6.0f};
            input.at(3) = {7.0f, 8.0f};

            ctx.setInput("padded", input);

            REQUIRE(ctx.run() == Result::SUCCESS);

            auto& outUnpadded = ctx.output("unpadded");
            auto& outPad = ctx.output("pad");

            REQUIRE(outUnpadded.shape().size() == 1);
            REQUIRE(outUnpadded.shape(0) == 2);

            REQUIRE_THAT(outUnpadded.at<CF32>(0).real(), Catch::Matchers::WithinAbs(1.0f, 1e-6f));
            REQUIRE_THAT(outUnpadded.at<CF32>(0).imag(), Catch::Matchers::WithinAbs(2.0f, 1e-6f));
            REQUIRE_THAT(outUnpadded.at<CF32>(1).real(), Catch::Matchers::WithinAbs(3.0f, 1e-6f));
            REQUIRE_THAT(outUnpadded.at<CF32>(1).imag(), Catch::Matchers::WithinAbs(4.0f, 1e-6f));

            REQUIRE(outPad.shape().size() == 1);
            REQUIRE(outPad.shape(0) == 2);

            REQUIRE_THAT(outPad.at<CF32>(0).real(), Catch::Matchers::WithinAbs(5.0f, 1e-6f));
            REQUIRE_THAT(outPad.at<CF32>(0).imag(), Catch::Matchers::WithinAbs(6.0f, 1e-6f));
            REQUIRE_THAT(outPad.at<CF32>(1).real(), Catch::Matchers::WithinAbs(7.0f, 1e-6f));
            REQUIRE_THAT(outPad.at<CF32>(1).imag(), Catch::Matchers::WithinAbs(8.0f, 1e-6f));
        }
    }
}

TEST_CASE("Unpad Module - Full Unpad F32", "[modules][unpad][F32]") {
    auto implementations = Registry::ListAvailableModules("unpad");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            TestContext ctx("unpad", impl.device, impl.runtime, impl.provider);

            Modules::Unpad config;
            config.size = 4;
            config.axis = 0;

            ctx.setConfig(config);

            auto input = ctx.createTensor<F32>({4});
            for (U64 i = 0; i < 4; ++i) {
                input.at(i) = static_cast<F32>(i + 1);
            }

            ctx.setInput("padded", input);

            REQUIRE(ctx.run() == Result::SUCCESS);

            auto& outUnpadded = ctx.output("unpadded");
            auto& outPad = ctx.output("pad");

            // Unpadded should be empty.
            REQUIRE(outUnpadded.shape().size() == 1);
            REQUIRE(outUnpadded.shape(0) == 0);

            // Pad should have all elements.
            REQUIRE(outPad.shape().size() == 1);
            REQUIRE(outPad.shape(0) == 4);

            for (U64 i = 0; i < 4; ++i) {
                REQUIRE_THAT(outPad.at<F32>(i),
                             Catch::Matchers::WithinAbs(static_cast<F32>(i + 1), 1e-6f));
            }
        }
    }
}

TEST_CASE("Unpad Module - Validation rejects invalid axis and size",
          "[modules][unpad][validation]") {
    auto implementations = Registry::ListAvailableModules("unpad");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        SECTION("axis out of range") {
            TestContext ctx("unpad", impl.device, impl.runtime, impl.provider);

            Modules::Unpad config;
            config.size = 1;
            config.axis = 3;
            ctx.setConfig(config);

            auto input = ctx.createTensor<F32>({2, 3});
            ctx.setInput("padded", input);

            REQUIRE(ctx.run() == Result::ERROR);
        }

        SECTION("size larger than axis dimension") {
            TestContext ctx("unpad", impl.device, impl.runtime, impl.provider);

            Modules::Unpad config;
            config.size = 5;
            config.axis = 0;
            ctx.setConfig(config);

            auto input = ctx.createTensor<F32>({4});
            ctx.setInput("padded", input);

            REQUIRE(ctx.run() == Result::ERROR);
        }
    }
}
