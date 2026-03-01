#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "jetstream/testing.hh"
#include "jetstream/registry.hh"
#include "jetstream/domains/dsp/am/module.hh"

#include <cmath>

using namespace Jetstream;

TEST_CASE("AM - Constant Envelope Input", "[modules][am]") {
    auto implementations = Registry::ListAvailableModules("am");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device
                        << " Runtime: " << impl.runtime) {
            TestContext ctx("am", impl.device, impl.runtime,
                           impl.provider);

            Modules::AM config;
            config.sampleRate = 240e3f;
            config.dcAlpha = 0.995f;

            ctx.setConfig(config);

            // Create constant envelope input.
            const U64 bufferSize = 1024;
            Tensor input;
            REQUIRE(input.create(DeviceType::CPU, DataType::CF32,
                                 {bufferSize}) == Result::SUCCESS);

            for (U64 i = 0; i < bufferSize; ++i) {
                input.at<CF32>(i) = CF32(1.0f, 0.0f);
            }

            ctx.setInput("signal", input);

            REQUIRE(ctx.run() == Result::SUCCESS);

            auto& out = ctx.output("signal");

            // With constant envelope and DC blocker, output should
            // converge toward zero after initial transient.
            const F32 lastSample = out.at<F32>(bufferSize - 1);
            REQUIRE_THAT(lastSample,
                         Catch::Matchers::WithinAbs(0.0f, 0.1f));
        }
    }
}

TEST_CASE("AM - Modulated Signal", "[modules][am][modulation]") {
    auto implementations = Registry::ListAvailableModules("am");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device
                        << " Runtime: " << impl.runtime) {
            TestContext ctx("am", impl.device, impl.runtime,
                           impl.provider);

            const F32 sampleRate = 240e3f;
            Modules::AM config;
            config.sampleRate = sampleRate;
            config.dcAlpha = 0.995f;

            ctx.setConfig(config);

            // Create AM modulated signal:
            // carrier with amplitude modulated by a low-frequency tone.
            const U64 bufferSize = 2048;
            const F32 carrierFreq = 10e3f;
            const F32 modFreq = 1e3f;
            const F32 modIndex = 0.5f;

            Tensor input;
            REQUIRE(input.create(DeviceType::CPU, DataType::CF32,
                                 {bufferSize}) == Result::SUCCESS);

            for (U64 i = 0; i < bufferSize; ++i) {
                const F32 t = static_cast<F32>(i) / sampleRate;
                const F32 mod = 1.0f
                    + modIndex * std::cos(2.0f * JST_PI * modFreq * t);
                const F32 phase = 2.0f * JST_PI * carrierFreq * t;
                input.at<CF32>(i) = CF32(
                    mod * std::cos(phase),
                    mod * std::sin(phase));
            }

            ctx.setInput("signal", input);

            REQUIRE(ctx.run() == Result::SUCCESS);

            auto& out = ctx.output("signal");

            // Output should contain the modulation frequency.
            // Verify output is not all zeros (has variation).
            F32 minVal = out.at<F32>(0);
            F32 maxVal = out.at<F32>(0);
            for (U64 i = 1; i < bufferSize; ++i) {
                minVal = std::min(minVal, out.at<F32>(i));
                maxVal = std::max(maxVal, out.at<F32>(i));
            }
            const F32 range = maxVal - minVal;
            REQUIRE(range > 0.01f);
        }
    }
}

TEST_CASE("AM - Output Size Matches Input", "[modules][am][size]") {
    auto implementations = Registry::ListAvailableModules("am");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device
                        << " Runtime: " << impl.runtime) {
            TestContext ctx("am", impl.device, impl.runtime,
                           impl.provider);

            Modules::AM config;
            config.sampleRate = 240e3f;
            config.dcAlpha = 0.995f;

            ctx.setConfig(config);

            const U64 bufferSize = 1024;
            Tensor input;
            REQUIRE(input.create(DeviceType::CPU, DataType::CF32,
                                 {bufferSize}) == Result::SUCCESS);

            for (U64 i = 0; i < bufferSize; ++i) {
                input.at<CF32>(i) = CF32(1.0f, 0.0f);
            }

            ctx.setInput("signal", input);

            REQUIRE(ctx.run() == Result::SUCCESS);

            auto& out = ctx.output("signal");

            REQUIRE(out.size() == bufferSize);
        }
    }
}
