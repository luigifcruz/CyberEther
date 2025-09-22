#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "jetstream/testing.hh"
#include "jetstream/registry.hh"
#include "jetstream/domains/dsp/rrc_filter/module.hh"

#include <cmath>

using namespace Jetstream;

TEST_CASE("RRC Filter - CF32 Impulse Response",
          "[modules][rrc_filter][cf32]") {
    auto implementations =
        Registry::ListAvailableModules("rrc_filter");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device
                        << " Runtime: " << impl.runtime) {
            TestContext ctx("rrc_filter", impl.device,
                           impl.runtime, impl.provider);

            Modules::RrcFilter config;
            config.symbolRate = 1.0e6f;
            config.sampleRate = 4.0e6f;
            config.rollOff = 0.35f;
            config.taps = 11;

            ctx.setConfig(config);

            // Create impulse input: 1 at index 0, zeros elsewhere.
            const U64 bufferSize = 32;
            Tensor input;
            REQUIRE(input.create(DeviceType::CPU, DataType::CF32,
                                 {bufferSize}) == Result::SUCCESS);

            for (U64 i = 0; i < bufferSize; ++i) {
                input.at<CF32>(i) = CF32(0.0f, 0.0f);
            }
            input.at<CF32>(0) = CF32(1.0f, 0.0f);

            ctx.setInput("buffer", input);

            REQUIRE(ctx.run() == Result::SUCCESS);

            auto& out = ctx.output("buffer");

            // The impulse response should be the filter
            // coefficients (delayed by taps/2). The peak should be
            // near the center of the filter at index taps/2 = 5.
            // Verify that the output is not all zeros.
            F32 maxMag = 0.0f;
            U64 maxIdx = 0;
            for (U64 i = 0; i < bufferSize; ++i) {
                F32 mag = std::abs(out.at<CF32>(i));
                if (mag > maxMag) {
                    maxMag = mag;
                    maxIdx = i;
                }
            }

            // Peak should be at index taps/2 = 5.
            REQUIRE(maxIdx == config.taps / 2);
            REQUIRE(maxMag > 0.0f);
        }
    }
}

TEST_CASE("RRC Filter - F32 DC Passthrough",
          "[modules][rrc_filter][f32]") {
    auto implementations =
        Registry::ListAvailableModules("rrc_filter");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device
                        << " Runtime: " << impl.runtime) {
            TestContext ctx("rrc_filter", impl.device,
                           impl.runtime, impl.provider);

            Modules::RrcFilter config;
            config.symbolRate = 1.0e6f;
            config.sampleRate = 4.0e6f;
            config.rollOff = 0.35f;
            config.taps = 11;

            ctx.setConfig(config);

            // Create constant DC input.
            const U64 bufferSize = 64;
            Tensor input;
            REQUIRE(input.create(DeviceType::CPU, DataType::F32,
                                 {bufferSize}) == Result::SUCCESS);

            for (U64 i = 0; i < bufferSize; ++i) {
                input.at<F32>(i) = 1.0f;
            }

            ctx.setInput("buffer", input);

            REQUIRE(ctx.run() == Result::SUCCESS);

            auto& out = ctx.output("buffer");

            // After the filter settles (past taps-1 samples),
            // the output should converge to the sum of all
            // coefficients times the DC value.
            // Verify the tail samples are approximately equal.
            const F32 tailValue = out.at<F32>(bufferSize - 1);
            for (U64 i = bufferSize - 10; i < bufferSize; ++i) {
                REQUIRE_THAT(out.at<F32>(i),
                    Catch::Matchers::WithinRel(tailValue, 0.01f));
            }
        }
    }
}

TEST_CASE("RRC Filter - Invalid Even Taps",
          "[modules][rrc_filter][error]") {
    auto implementations =
        Registry::ListAvailableModules("rrc_filter");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device
                        << " Runtime: " << impl.runtime) {
            TestContext ctx("rrc_filter", impl.device,
                           impl.runtime, impl.provider);

            Modules::RrcFilter config;
            config.taps = 10;  // Even number, should fail.

            ctx.setConfig(config);

            Tensor input;
            REQUIRE(input.create(DeviceType::CPU, DataType::CF32,
                                 {64}) == Result::SUCCESS);

            ctx.setInput("buffer", input);

            REQUIRE(ctx.run() != Result::SUCCESS);
        }
    }
}

TEST_CASE("RRC Filter - Invalid Sample Rate",
          "[modules][rrc_filter][error]") {
    auto implementations =
        Registry::ListAvailableModules("rrc_filter");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device
                        << " Runtime: " << impl.runtime) {
            TestContext ctx("rrc_filter", impl.device,
                           impl.runtime, impl.provider);

            Modules::RrcFilter config;
            config.symbolRate = 2.0e6f;
            config.sampleRate = 1.0e6f;  // Less than symbol rate.

            ctx.setConfig(config);

            Tensor input;
            REQUIRE(input.create(DeviceType::CPU, DataType::CF32,
                                 {64}) == Result::SUCCESS);

            ctx.setInput("buffer", input);

            REQUIRE(ctx.run() != Result::SUCCESS);
        }
    }
}
