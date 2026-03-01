#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "jetstream/testing.hh"
#include "jetstream/registry.hh"
#include "jetstream/domains/dsp/psk_demod/module.hh"

#include <cmath>

using namespace Jetstream;

TEST_CASE("PskDemod - Output Size Decimation", "[modules][psk_demod]") {
    auto implementations = Registry::ListAvailableModules("psk_demod");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            TestContext ctx("psk_demod", impl.device, impl.runtime, impl.provider);

            Modules::PskDemod config;
            config.pskType = "qpsk";
            config.sampleRate = 2000000.0;
            config.symbolRate = 500000.0;

            ctx.setConfig(config);

            const U64 inputSize = 8192;
            const U64 expectedOutputSize = inputSize / 4;  // sampleRate / symbolRate = 4

            Tensor input;
            REQUIRE(input.create(DeviceType::CPU, DataType::CF32, {inputSize}) == Result::SUCCESS);

            for (U64 i = 0; i < inputSize; ++i) {
                input.at<CF32>(i) = CF32(1.0f, 0.0f);
            }

            ctx.setInput("signal", input);

            REQUIRE(ctx.run() == Result::SUCCESS);

            auto& out = ctx.output("signal");

            REQUIRE(out.size() == expectedOutputSize);
        }
    }
}

TEST_CASE("PskDemod - BPSK Constant Phase", "[modules][psk_demod][bpsk]") {
    auto implementations = Registry::ListAvailableModules("psk_demod");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            TestContext ctx("psk_demod", impl.device, impl.runtime, impl.provider);

            Modules::PskDemod config;
            config.pskType = "bpsk";
            config.sampleRate = 1000000.0;
            config.symbolRate = 250000.0;
            config.frequencyLoopBandwidth = 0.01;
            config.timingLoopBandwidth = 0.01;
            config.dampingFactor = 0.707;

            ctx.setConfig(config);

            const U64 inputSize = 1024;
            Tensor input;
            REQUIRE(input.create(DeviceType::CPU, DataType::CF32, {inputSize}) == Result::SUCCESS);

            // Create constant positive phase BPSK signal.
            for (U64 i = 0; i < inputSize; ++i) {
                input.at<CF32>(i) = CF32(1.0f, 0.0f);
            }

            ctx.setInput("signal", input);

            REQUIRE(ctx.run() == Result::SUCCESS);

            auto& out = ctx.output("signal");

            // After initial transient, output should be near positive real axis.
            for (U64 i = out.size() / 2; i < out.size(); ++i) {
                REQUIRE(out.at<CF32>(i).real() > 0.0f);
            }
        }
    }
}

TEST_CASE("PskDemod - QPSK Quadrants", "[modules][psk_demod][qpsk]") {
    auto implementations = Registry::ListAvailableModules("psk_demod");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            TestContext ctx("psk_demod", impl.device, impl.runtime, impl.provider);

            Modules::PskDemod config;
            config.pskType = "qpsk";
            config.sampleRate = 2000000.0;
            config.symbolRate = 500000.0;
            config.frequencyLoopBandwidth = 0.05;
            config.timingLoopBandwidth = 0.05;
            config.dampingFactor = 0.707;

            ctx.setConfig(config);

            const U64 inputSize = 4096;
            Tensor input;
            REQUIRE(input.create(DeviceType::CPU, DataType::CF32, {inputSize}) == Result::SUCCESS);

            // Create first quadrant QPSK signal.
            constexpr F32 INV_SQRT2 = 0.7071067811865475f;
            for (U64 i = 0; i < inputSize; ++i) {
                input.at<CF32>(i) = CF32(INV_SQRT2, INV_SQRT2);
            }

            ctx.setInput("signal", input);

            REQUIRE(ctx.run() == Result::SUCCESS);

            auto& out = ctx.output("signal");

            // After transient, output should be in first quadrant.
            for (U64 i = out.size() / 2; i < out.size(); ++i) {
                REQUIRE(out.at<CF32>(i).real() > 0.0f);
                REQUIRE(out.at<CF32>(i).imag() > 0.0f);
            }
        }
    }
}

TEST_CASE("PskDemod - 8PSK Basic", "[modules][psk_demod][8psk]") {
    auto implementations = Registry::ListAvailableModules("psk_demod");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            TestContext ctx("psk_demod", impl.device, impl.runtime, impl.provider);

            Modules::PskDemod config;
            config.pskType = "8psk";
            config.sampleRate = 4000000.0;
            config.symbolRate = 1000000.0;
            config.frequencyLoopBandwidth = 0.05;
            config.timingLoopBandwidth = 0.05;
            config.dampingFactor = 0.707;

            ctx.setConfig(config);

            const U64 inputSize = 8192;
            Tensor input;
            REQUIRE(input.create(DeviceType::CPU, DataType::CF32, {inputSize}) == Result::SUCCESS);

            // Create 8-PSK signal at 0 degrees.
            for (U64 i = 0; i < inputSize; ++i) {
                input.at<CF32>(i) = CF32(1.0f, 0.0f);
            }

            ctx.setInput("signal", input);

            REQUIRE(ctx.run() == Result::SUCCESS);

            auto& out = ctx.output("signal");

            // Check output is reasonable (non-zero magnitude).
            for (U64 i = out.size() / 2; i < out.size(); ++i) {
                const F32 mag = std::abs(out.at<CF32>(i));
                REQUIRE(mag > 0.1f);
            }
        }
    }
}

TEST_CASE("PskDemod - Invalid Configuration", "[modules][psk_demod][validation]") {
    auto implementations = Registry::ListAvailableModules("psk_demod");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            SECTION("Symbol rate greater than sample rate") {
                TestContext ctx("psk_demod", impl.device, impl.runtime, impl.provider);

                Modules::PskDemod config;
                config.pskType = "qpsk";
                config.sampleRate = 1000000.0;
                config.symbolRate = 2000000.0;

                ctx.setConfig(config);

                Tensor input;
                REQUIRE(input.create(DeviceType::CPU, DataType::CF32, {1024}) == Result::SUCCESS);

                ctx.setInput("signal", input);

                REQUIRE(ctx.run() == Result::ERROR);
            }

            SECTION("Negative sample rate") {
                TestContext ctx("psk_demod", impl.device, impl.runtime, impl.provider);

                Modules::PskDemod config;
                config.pskType = "qpsk";
                config.sampleRate = -1000000.0;
                config.symbolRate = 250000.0;

                ctx.setConfig(config);

                Tensor input;
                REQUIRE(input.create(DeviceType::CPU, DataType::CF32, {1024}) == Result::SUCCESS);

                ctx.setInput("signal", input);

                REQUIRE(ctx.run() == Result::ERROR);
            }

            SECTION("Invalid loop bandwidth") {
                TestContext ctx("psk_demod", impl.device, impl.runtime, impl.provider);

                Modules::PskDemod config;
                config.pskType = "qpsk";
                config.sampleRate = 2000000.0;
                config.symbolRate = 500000.0;
                config.frequencyLoopBandwidth = 1.5;

                ctx.setConfig(config);

                Tensor input;
                REQUIRE(input.create(DeviceType::CPU, DataType::CF32, {1024}) == Result::SUCCESS);

                ctx.setInput("signal", input);

                REQUIRE(ctx.run() == Result::ERROR);
            }
        }
    }
}
