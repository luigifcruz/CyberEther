#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "jetstream/testing.hh"
#include "jetstream/registry.hh"
#include "jetstream/domains/dsp/amplitude/module.hh"

#include <cmath>

using namespace Jetstream;

TEST_CASE("Amplitude - CF32 DC Signal", "[modules][amplitude][cf32]") {
    auto implementations = Registry::ListAvailableModules("amplitude");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            TestContext ctx("amplitude", impl.device, impl.runtime, impl.provider);

            Modules::Amplitude config;

            ctx.setConfig(config);

            // Create a constant complex signal.
            const U64 bufferSize = 64;
            Tensor input;
            REQUIRE(input.create(DeviceType::CPU, DataType::CF32, {bufferSize}) == Result::SUCCESS);

            const F32 magnitude = 1.0f;
            for (U64 i = 0; i < bufferSize; ++i) {
                input.at<CF32>(i) = CF32(magnitude, 0.0f);
            }

            ctx.setInput("signal", input);

            REQUIRE(ctx.run() == Result::SUCCESS);

            auto& out = ctx.output("signal");

            // All output values should be the same (constant input).
            // Expected: 20*log10(1.0) + 20*log10(1/64) = 0 + (-36.12) ≈ -36.12 dB
            const F32 scalingCoeff = 20.0f * std::log10(1.0f / static_cast<F32>(bufferSize));
            const F32 expected = 20.0f * std::log10(magnitude) + scalingCoeff;

            for (U64 i = 0; i < bufferSize; ++i) {
                REQUIRE_THAT(out.at<F32>(i), Catch::Matchers::WithinAbs(expected, 0.5f));
            }
        }
    }
}

TEST_CASE("Amplitude - F32 Signal", "[modules][amplitude][f32]") {
    auto implementations = Registry::ListAvailableModules("amplitude");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            TestContext ctx("amplitude", impl.device, impl.runtime, impl.provider);

            Modules::Amplitude config;

            ctx.setConfig(config);

            // Create a constant real signal.
            const U64 bufferSize = 128;
            Tensor input;
            REQUIRE(input.create(DeviceType::CPU, DataType::F32, {bufferSize}) == Result::SUCCESS);

            const F32 value = 2.0f;
            for (U64 i = 0; i < bufferSize; ++i) {
                input.at<F32>(i) = value;
            }

            ctx.setInput("signal", input);

            REQUIRE(ctx.run() == Result::SUCCESS);

            auto& out = ctx.output("signal");

            // Expected: 20*log10(2.0) + 20*log10(1/128) ≈ 6.02 + (-42.14) ≈ -36.12 dB
            const F32 scalingCoeff = 20.0f * std::log10(1.0f / static_cast<F32>(bufferSize));
            const F32 expected = 20.0f * std::log10(value) + scalingCoeff;

            for (U64 i = 0; i < bufferSize; ++i) {
                REQUIRE_THAT(out.at<F32>(i), Catch::Matchers::WithinAbs(expected, 0.5f));
            }
        }
    }
}

TEST_CASE("Amplitude - CF32 Various Magnitudes", "[modules][amplitude][magnitude]") {
    auto implementations = Registry::ListAvailableModules("amplitude");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            TestContext ctx("amplitude", impl.device, impl.runtime, impl.provider);

            Modules::Amplitude config;

            ctx.setConfig(config);

            const U64 bufferSize = 4;
            Tensor input;
            REQUIRE(input.create(DeviceType::CPU, DataType::CF32, {bufferSize}) == Result::SUCCESS);

            // Create complex samples with known magnitudes: 1, 2, 3, 4
            input.at<CF32>(0) = CF32(1.0f, 0.0f);   // magnitude = 1
            input.at<CF32>(1) = CF32(0.0f, 2.0f);   // magnitude = 2
            input.at<CF32>(2) = CF32(2.4f, 1.8f);   // magnitude = 3 (3-4-5 triangle scaled)
            input.at<CF32>(3) = CF32(4.0f, 0.0f);   // magnitude = 4

            ctx.setInput("signal", input);

            REQUIRE(ctx.run() == Result::SUCCESS);

            auto& out = ctx.output("signal");

            const F32 scalingCoeff = 20.0f * std::log10(1.0f / static_cast<F32>(bufferSize));

            // Verify each output corresponds to 20*log10(magnitude) + scalingCoeff
            REQUIRE_THAT(out.at<F32>(0), Catch::Matchers::WithinAbs(20.0f * std::log10(1.0f) + scalingCoeff, 0.5f));
            REQUIRE_THAT(out.at<F32>(1), Catch::Matchers::WithinAbs(20.0f * std::log10(2.0f) + scalingCoeff, 0.5f));
            REQUIRE_THAT(out.at<F32>(2), Catch::Matchers::WithinAbs(20.0f * std::log10(3.0f) + scalingCoeff, 0.5f));
            REQUIRE_THAT(out.at<F32>(3), Catch::Matchers::WithinAbs(20.0f * std::log10(4.0f) + scalingCoeff, 0.5f));
        }
    }
}
