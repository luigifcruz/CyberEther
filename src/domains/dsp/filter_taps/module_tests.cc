#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "jetstream/testing.hh"
#include "jetstream/registry.hh"
#include "jetstream/domains/dsp/filter_taps/module.hh"

#include <cmath>
#include <complex>

using namespace Jetstream;

TEST_CASE("Filter Taps - Default Config", "[modules][filter_taps]") {
    auto implementations = Registry::ListAvailableModules("filter_taps");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            TestContext ctx("filter_taps", impl.device, impl.runtime, impl.provider);

            Modules::FilterTaps config;
            config.sampleRate = 2.0e6;
            config.bandwidth = 1.0e6;
            config.center = {0.0};
            config.taps = 101;

            ctx.setConfig(config);

            REQUIRE(ctx.run() == Result::SUCCESS);

            auto& out = ctx.output("coeffs");
            REQUIRE(out.size() == 101);
            REQUIRE(out.dtype() == DataType::CF32);

            // Center tap should be the maximum magnitude.
            const U64 centerIdx = 50;
            F32 centerMag = std::abs(out.at<CF32>(0, centerIdx));
            for (U64 i = 0; i < 101; ++i) {
                REQUIRE(std::abs(out.at<CF32>(0, i)) <= centerMag + 1e-6f);
            }
        }
    }
}

TEST_CASE("Filter Taps - Zero Center Symmetry", "[modules][filter_taps]") {
    auto implementations = Registry::ListAvailableModules("filter_taps");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            TestContext ctx("filter_taps", impl.device, impl.runtime, impl.provider);

            Modules::FilterTaps config;
            config.sampleRate = 2.0e6;
            config.bandwidth = 0.5e6;
            config.center = {0.0};
            config.taps = 51;

            ctx.setConfig(config);

            REQUIRE(ctx.run() == Result::SUCCESS);

            auto& out = ctx.output("coeffs");

            // With zero center, coefficients should be real-valued (imaginary ~0).
            for (U64 i = 0; i < config.taps; ++i) {
                REQUIRE_THAT(out.at<CF32>(0, i).imag(),
                             Catch::Matchers::WithinAbs(0.0f, 1e-6f));
            }

            // Coefficients should be symmetric around center.
            const U64 center = config.taps / 2;
            for (U64 i = 0; i < center; ++i) {
                REQUIRE_THAT(out.at<CF32>(0, i).real(),
                             Catch::Matchers::WithinAbs(
                                 out.at<CF32>(0, config.taps - 1 - i).real(),
                                 1e-6f));
            }
        }
    }
}

TEST_CASE("Filter Taps - Tensor Attributes", "[modules][filter_taps]") {
    auto implementations = Registry::ListAvailableModules("filter_taps");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            TestContext ctx("filter_taps", impl.device, impl.runtime, impl.provider);

            Modules::FilterTaps config;
            config.sampleRate = 2.0e6;
            config.bandwidth = 0.5e6;
            config.center = {0.1e6};
            config.taps = 51;

            ctx.setConfig(config);

            REQUIRE(ctx.run() == Result::SUCCESS);

            auto& out = ctx.output("coeffs");

            REQUIRE(out.hasAttribute("sampleRate"));
            REQUIRE(out.hasAttribute("bandwidth"));
            REQUIRE(out.hasAttribute("center"));

            REQUIRE_THAT(std::any_cast<F32>(out.attribute("sampleRate")),
                         Catch::Matchers::WithinAbs(2.0e6f, 1.0f));
            REQUIRE_THAT(std::any_cast<F32>(out.attribute("bandwidth")),
                         Catch::Matchers::WithinAbs(0.5e6f, 1.0f));
            REQUIRE_THAT(std::any_cast<F32>(out.attribute("center")),
                         Catch::Matchers::WithinAbs(0.1e6f, 1.0f));
        }
    }
}

TEST_CASE("Filter Taps - Multi-Head", "[modules][filter_taps]") {
    auto implementations = Registry::ListAvailableModules("filter_taps");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            TestContext ctx("filter_taps", impl.device, impl.runtime, impl.provider);

            Modules::FilterTaps config;
            config.sampleRate = 2.0e6;
            config.bandwidth = 0.2e6;
            config.center = {0.0, 0.2e6, -0.4e6};
            config.taps = 51;

            ctx.setConfig(config);

            REQUIRE(ctx.run() == Result::SUCCESS);

            auto& out = ctx.output("coeffs");

            // Output should be 2D: {3, 51}.
            REQUIRE(out.rank() == 2);
            REQUIRE(out.shape(0) == 3);
            REQUIRE(out.shape(1) == 51);

            // First head (center=0) should have real-valued coefficients.
            for (U64 i = 0; i < config.taps; ++i) {
                REQUIRE_THAT(out.at<CF32>(0, i).imag(),
                             Catch::Matchers::WithinAbs(0.0f, 1e-6f));
            }

            // Second head (center=0.2MHz) should have complex coefficients.
            bool hasComplex = false;
            for (U64 i = 0; i < config.taps; ++i) {
                if (std::abs(out.at<CF32>(1, i).imag()) > 1e-6f) {
                    hasComplex = true;
                    break;
                }
            }
            REQUIRE(hasComplex);
        }
    }
}

TEST_CASE("Filter Taps - Center Tap Matches Normalized Bandwidth", "[modules][filter_taps]") {
    auto implementations = Registry::ListAvailableModules("filter_taps");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            TestContext ctx("filter_taps", impl.device, impl.runtime, impl.provider);

            Modules::FilterTaps config;
            config.sampleRate = 2.0e6;
            config.bandwidth = 0.2e6;
            config.center = {0.0};
            config.taps = 101;

            ctx.setConfig(config);

            REQUIRE(ctx.run() == Result::SUCCESS);

            auto& out = ctx.output("coeffs");
            const U64 centerIdx = config.taps / 2;

            const F32 expectedCenter = static_cast<F32>(config.bandwidth / config.sampleRate);
            REQUIRE_THAT(out.at<CF32>(0, centerIdx).real(),
                         Catch::Matchers::WithinRel(expectedCenter, 1e-4f));
            REQUIRE_THAT(out.at<CF32>(0, centerIdx).imag(),
                         Catch::Matchers::WithinAbs(0.0f, 1e-6f));
        }
    }
}
