#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>
#include <limits>

#include "jetstream/domains/core/range/module.hh"
#include "jetstream/registry.hh"
#include "jetstream/testing.hh"

using namespace Jetstream;

namespace {

F32 SoftRange(const F32 value, const F32 min, const F32 max) {
    const F32 normalized = (value - min) / (max - min);
    return 0.5f + 0.5f * std::tanh(4.0f * (normalized - 0.5f));
}

}  // namespace

TEST_CASE("Range Module - Scales Into Unit Interval", "[modules][range][F32]") {
    const auto implementations = Registry::ListAvailableModules("range");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            TestContext ctx("range", impl.device, impl.runtime, impl.provider);

            Modules::Range config;
            config.min = -2.0f;
            config.max = 2.0f;
            ctx.setConfig(config);

            auto input = ctx.createTensor<F32>({3});
            input.at(0) = -2.0f;
            input.at(1) = 0.0f;
            input.at(2) = 2.0f;

            ctx.setInput("signal", input);
            REQUIRE(ctx.run() == Result::SUCCESS);

            auto& out = ctx.output("signal");
            REQUIRE_THAT(out.at<F32>(0),
                         Catch::Matchers::WithinAbs(SoftRange(-2.0f, -2.0f, 2.0f), 1e-6f));
            REQUIRE_THAT(out.at<F32>(1), Catch::Matchers::WithinAbs(0.5f, 1e-6f));
            REQUIRE_THAT(out.at<F32>(2),
                         Catch::Matchers::WithinAbs(SoftRange(2.0f, -2.0f, 2.0f), 1e-6f));
        }
    }
}

TEST_CASE("Range Module - Softly Compresses Outliers",
          "[modules][range][F32][soft-knee]") {
    const auto implementations = Registry::ListAvailableModules("range");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            TestContext ctx("range", impl.device, impl.runtime, impl.provider);

            Modules::Range config;
            config.min = -2.0f;
            config.max = 2.0f;
            ctx.setConfig(config);

            auto input = ctx.createTensor<F32>({4});
            input.at(0) = -std::numeric_limits<F32>::infinity();
            input.at(1) = -4.0f;
            input.at(2) = 4.0f;
            input.at(3) = std::numeric_limits<F32>::infinity();

            ctx.setInput("signal", input);
            REQUIRE(ctx.run() == Result::SUCCESS);

            const auto& out = ctx.output("signal");
            REQUIRE(out.at<F32>(0) == 0.0f);
            REQUIRE_THAT(out.at<F32>(1),
                         Catch::Matchers::WithinAbs(SoftRange(-4.0f, -2.0f, 2.0f), 1e-6f));
            REQUIRE_THAT(out.at<F32>(2),
                         Catch::Matchers::WithinAbs(SoftRange(4.0f, -2.0f, 2.0f), 1e-6f));
            REQUIRE(out.at<F32>(3) == 1.0f);
            REQUIRE(out.at<F32>(1) > 0.0f);
            REQUIRE(out.at<F32>(2) < 1.0f);
        }
    }
}

TEST_CASE("Range Module - Collapses Equal Bounds To Midpoint",
          "[modules][range][F32][equal-bounds]") {
    const auto implementations = Registry::ListAvailableModules("range");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            TestContext ctx("range", impl.device, impl.runtime, impl.provider);

            Modules::Range config;
            config.min = 1.0f;
            config.max = 1.0f;
            ctx.setConfig(config);

            auto input = ctx.createTensor<F32>({4});
            input.at(0) = -std::numeric_limits<F32>::infinity();
            input.at(1) = -100.0f;
            input.at(2) = 100.0f;
            input.at(3) = std::numeric_limits<F32>::infinity();
            ctx.setInput("signal", input);
            REQUIRE(ctx.run() == Result::SUCCESS);

            const auto& out = ctx.output("signal");
            REQUIRE(out.at<F32>(0) == 0.5f);
            REQUIRE(out.at<F32>(1) == 0.5f);
            REQUIRE(out.at<F32>(2) == 0.5f);
            REQUIRE(out.at<F32>(3) == 0.5f);
        }
    }
}

TEST_CASE("Range Module - Orders Reversed Bounds", "[modules][range][F32][reversed-bounds]") {
    const auto implementations = Registry::ListAvailableModules("range");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            TestContext ctx("range", impl.device, impl.runtime, impl.provider);

            Modules::Range config;
            config.min = 1.0f;
            config.max = -1.0f;
            ctx.setConfig(config);

            auto input = ctx.createTensor<F32>({3});
            input.at(0) = -1.0f;
            input.at(1) = 0.0f;
            input.at(2) = 1.0f;
            ctx.setInput("signal", input);
            REQUIRE(ctx.run() == Result::SUCCESS);

            const auto& out = ctx.output("signal");
            REQUIRE_THAT(out.at<F32>(0),
                         Catch::Matchers::WithinAbs(SoftRange(-1.0f, -1.0f, 1.0f), 1e-6f));
            REQUIRE(out.at<F32>(1) == 0.5f);
            REQUIRE_THAT(out.at<F32>(2),
                         Catch::Matchers::WithinAbs(SoftRange(1.0f, -1.0f, 1.0f), 1e-6f));
        }
    }
}

TEST_CASE("Range Module - Rank 4 Non-Contiguous",
          "[modules][range][F32][noncontiguous]") {
    const auto implementations = Registry::ListAvailableModules("range");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            TestContext ctx("range", impl.device, impl.runtime, impl.provider);

            Modules::Range config;
            config.min = 0.0f;
            config.max = 100.0f;
            ctx.setConfig(config);

            Tensor storage(DeviceType::CPU, DataType::F32, {2, 2, 3, 2, 4});
            for (U64 i = 0; i < storage.size(); ++i) {
                storage.data<F32>()[i] = static_cast<F32>(i + 1);
            }

            Tensor input = storage.clone();
            REQUIRE(input.slice({Token(1), Token(), Token(), Token(), Token()}) == Result::SUCCESS);
            REQUIRE(input.permute({1, 0, 3, 2}) == Result::SUCCESS);
            REQUIRE(input.shape() == Shape{3, 2, 4, 2});
            REQUIRE(input.offset() != 0);
            REQUIRE_FALSE(input.contiguous());

            ctx.setInput("signal", input);
            REQUIRE(ctx.run() == Result::SUCCESS);

            const auto& out = ctx.output("signal");
            for (U64 i = 0; i < 3; ++i) {
                for (U64 j = 0; j < 2; ++j) {
                    for (U64 k = 0; k < 4; ++k) {
                        for (U64 l = 0; l < 2; ++l) {
                            const F32 expected = SoftRange(
                                input.at<F32>(i, j, k, l), 0.0f, 100.0f);
                            REQUIRE_THAT(out.at<F32>(i, j, k, l),
                                         Catch::Matchers::WithinAbs(expected, 1e-6f));
                        }
                    }
                }
            }
        }
    }
}
