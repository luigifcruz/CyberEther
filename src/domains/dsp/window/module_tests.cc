#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "jetstream/testing.hh"
#include "jetstream/registry.hh"
#include "jetstream/domains/dsp/window/module.hh"

#include <cmath>
#include <vector>

using namespace Jetstream;

TEST_CASE("Window - Blackman coefficients match formula", "[modules][window]") {
    auto implementations = Registry::ListAvailableModules("window");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            TestContext ctx("window", impl.device, impl.runtime, impl.provider);

            Modules::Window config;
            config.size = 64;

            ctx.setConfig(config);

            REQUIRE(ctx.run() == Result::SUCCESS);

            auto& out = ctx.output("window");
            REQUIRE(out.dtype() == DataType::CF32);
            REQUIRE(out.rank() == 1);
            REQUIRE(out.shape(0) == config.size);

            for (U64 i = 0; i < config.size; ++i) {
                const F32 expected = static_cast<F32>(
                    0.42 - 0.50 * std::cos(2.0 * JST_PI * i / (config.size - 1)) +
                    0.08 * std::cos(4.0 * JST_PI * i / (config.size - 1)));

                REQUIRE_THAT(out.at<CF32>(i).real(),
                             Catch::Matchers::WithinAbs(expected, 1e-5f));
                REQUIRE_THAT(out.at<CF32>(i).imag(),
                             Catch::Matchers::WithinAbs(0.0f, 1e-6f));
            }
        }
    }
}

TEST_CASE("Window - shape and value bounds for various sizes", "[modules][window][sizes]") {
    auto implementations = Registry::ListAvailableModules("window");
    REQUIRE(!implementations.empty());

    const std::vector<U64> testSizes = {2, 16, 128, 512, 2048};

    for (const auto& impl : implementations) {
        for (const auto& testSize : testSizes) {
            DYNAMIC_SECTION("Device: " << impl.device << " Size: " << testSize) {
                TestContext ctx("window", impl.device, impl.runtime, impl.provider);

                Modules::Window config;
                config.size = testSize;

                ctx.setConfig(config);

                REQUIRE(ctx.run() == Result::SUCCESS);

                auto& out = ctx.output("window");

                REQUIRE(out.size() == testSize);
                REQUIRE(out.dtype() == DataType::CF32);
                REQUIRE(out.rank() == 1);
                REQUIRE(out.shape(0) == testSize);

                for (U64 i = 0; i < testSize / 2; ++i) {
                    REQUIRE_THAT(out.at<CF32>(i).real(),
                                 Catch::Matchers::WithinAbs(
                                     out.at<CF32>(testSize - 1 - i).real(),
                                     1e-5f));
                }

                for (U64 i = 0; i < testSize; ++i) {
                    F32 val = out.at<CF32>(i).real();
                    REQUIRE(val >= -1e-6f);
                    REQUIRE(val <= 1.0f + 1e-6f);
                }
            }
        }
    }
}

TEST_CASE("Window - invalid size is rejected", "[modules][window][validation]") {
    auto implementations = Registry::ListAvailableModules("window");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            TestContext ctx("window", impl.device, impl.runtime, impl.provider);

            Modules::Window config;
            config.size = 0;
            ctx.setConfig(config);

            REQUIRE(ctx.run() == Result::ERROR);
        }
    }
}

TEST_CASE("Window - repeated run keeps baked output stable", "[modules][window][state]") {
    auto implementations = Registry::ListAvailableModules("window");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            TestContext ctx("window", impl.device, impl.runtime, impl.provider);

            Modules::Window config;
            config.size = 256;
            ctx.setConfig(config);

            REQUIRE(ctx.run() == Result::SUCCESS);
            auto& out = ctx.output("window");

            std::vector<CF32> firstPass;
            firstPass.reserve(config.size);
            for (U64 i = 0; i < config.size; ++i) {
                firstPass.push_back(out.at<CF32>(i));
            }

            REQUIRE(ctx.run() == Result::SUCCESS);

            for (U64 i = 0; i < config.size; ++i) {
                REQUIRE_THAT(out.at<CF32>(i).real(),
                             Catch::Matchers::WithinAbs(firstPass[i].real(), 1e-7f));
                REQUIRE_THAT(out.at<CF32>(i).imag(),
                             Catch::Matchers::WithinAbs(firstPass[i].imag(), 1e-7f));
            }
        }
    }
}
