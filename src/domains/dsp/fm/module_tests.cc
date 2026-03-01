#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "jetstream/testing.hh"
#include "jetstream/registry.hh"
#include "jetstream/domains/dsp/fm/module.hh"

#include <cmath>

using namespace Jetstream;

TEST_CASE("FM - Constant Phase Input", "[modules][fm]") {
    auto implementations = Registry::ListAvailableModules("fm");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            TestContext ctx("fm", impl.device, impl.runtime, impl.provider);

            Modules::FM config;
            config.sampleRate = 240e3f;

            ctx.setConfig(config);

            // Create constant phase input (no frequency deviation).
            const U64 bufferSize = 64;
            Tensor input;
            REQUIRE(input.create(DeviceType::CPU, DataType::CF32, {bufferSize}) == Result::SUCCESS);

            // Constant complex value means no phase change.
            for (U64 i = 0; i < bufferSize; ++i) {
                input.at<CF32>(i) = CF32(1.0f, 0.0f);
            }

            ctx.setInput("signal", input);

            REQUIRE(ctx.run() == Result::SUCCESS);

            auto& out = ctx.output("signal");

            // With constant phase, output should be near zero (no frequency deviation).
            for (U64 i = 1; i < bufferSize; ++i) {
                REQUIRE_THAT(out.at<F32>(i), Catch::Matchers::WithinAbs(0.0f, 1e-5f));
            }
        }
    }
}

TEST_CASE("FM - Linear Phase Ramp", "[modules][fm][phase]") {
    auto implementations = Registry::ListAvailableModules("fm");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            TestContext ctx("fm", impl.device, impl.runtime, impl.provider);

            const F32 sampleRate = 240e3f;
            Modules::FM config;
            config.sampleRate = sampleRate;

            ctx.setConfig(config);

            // Create linear phase ramp (constant frequency offset).
            const U64 bufferSize = 128;
            const F32 freqOffset = 10e3f;  // 10 kHz offset
            const F32 phaseIncrement = 2.0f * JST_PI * freqOffset / sampleRate;

            Tensor input;
            REQUIRE(input.create(DeviceType::CPU, DataType::CF32, {bufferSize}) == Result::SUCCESS);

            F32 phase = 0.0f;
            for (U64 i = 0; i < bufferSize; ++i) {
                input.at<CF32>(i) = CF32(std::cos(phase), std::sin(phase));
                phase += phaseIncrement;
            }

            ctx.setInput("signal", input);

            REQUIRE(ctx.run() == Result::SUCCESS);

            auto& out = ctx.output("signal");

            // Output should be constant (constant frequency = constant demodulated value).
            // Expected value based on FM demod formula.
            const F32 kf = 100e3f / sampleRate;
            const F32 ref = 1.0f / (2.0f * JST_PI * kf);
            const F32 expected = phaseIncrement * ref;

            for (U64 i = 1; i < bufferSize; ++i) {
                REQUIRE_THAT(out.at<F32>(i), Catch::Matchers::WithinAbs(expected, 0.01f));
            }
        }
    }
}

TEST_CASE("FM - Output Size Matches Input", "[modules][fm][size]") {
    auto implementations = Registry::ListAvailableModules("fm");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            TestContext ctx("fm", impl.device, impl.runtime, impl.provider);

            Modules::FM config;
            config.sampleRate = 240e3f;

            ctx.setConfig(config);

            const U64 bufferSize = 1024;
            Tensor input;
            REQUIRE(input.create(DeviceType::CPU, DataType::CF32, {bufferSize}) == Result::SUCCESS);

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
