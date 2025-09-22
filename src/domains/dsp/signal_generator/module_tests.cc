#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "jetstream/testing.hh"
#include "jetstream/registry.hh"
#include "jetstream/domains/dsp/signal_generator/module.hh"

#include <cmath>

using namespace Jetstream;

TEST_CASE("Signal Generator - DC F32", "[modules][signal_generator][dc]") {
    auto implementations = Registry::ListAvailableModules("signal_generator");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            TestContext ctx("signal_generator", impl.device, impl.runtime, impl.provider);

            Modules::SignalGenerator config;
            config.signalType = "dc";
            config.signalDataType = "F32";
            config.bufferSize = 64;
            config.amplitude = 2.5;
            config.dcOffset = 1.0;

            ctx.setConfig(config);

            REQUIRE(ctx.run() == Result::SUCCESS);

            auto& out = ctx.output("signal");
            const F32 expected = static_cast<F32>(config.amplitude + config.dcOffset);

            for (U64 i = 0; i < config.bufferSize; ++i) {
                REQUIRE_THAT(out.at<F32>(i), Catch::Matchers::WithinAbs(expected, 1e-6f));
            }
        }
    }
}

TEST_CASE("Signal Generator - DC CF32", "[modules][signal_generator][dc][CF32]") {
    auto implementations = Registry::ListAvailableModules("signal_generator");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            TestContext ctx("signal_generator", impl.device, impl.runtime,
                            impl.provider);

            Modules::SignalGenerator config;
            config.signalType = "dc";
            config.signalDataType = "CF32";
            config.bufferSize = 16;
            config.amplitude = 2.0;
            config.dcOffset = -0.5;
            ctx.setConfig(config);

            REQUIRE(ctx.run() == Result::SUCCESS);

            auto& out = ctx.output("signal");
            const F32 expected = static_cast<F32>(config.amplitude +
                                                  config.dcOffset);

            for (U64 i = 0; i < config.bufferSize; ++i) {
                REQUIRE_THAT(out.at<CF32>(i).real(),
                             Catch::Matchers::WithinAbs(expected, 1e-6f));
                REQUIRE_THAT(out.at<CF32>(i).imag(),
                             Catch::Matchers::WithinAbs(0.0f, 1e-6f));
            }
        }
    }
}

TEST_CASE("Signal Generator - Sine F32", "[modules][signal_generator][sine]") {
    auto implementations = Registry::ListAvailableModules("signal_generator");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            TestContext ctx("signal_generator", impl.device, impl.runtime, impl.provider);

            Modules::SignalGenerator config;
            config.signalType = "sine";
            config.signalDataType = "F32";
            config.bufferSize = 128;
            config.sampleRate = 1000.0;
            config.frequency = 100.0;
            config.amplitude = 1.0;
            config.phase = 0.0;
            config.dcOffset = 0.0;

            ctx.setConfig(config);

            REQUIRE(ctx.run() == Result::SUCCESS);

            auto& out = ctx.output("signal");
            const F64 dt = 1.0 / config.sampleRate;

            for (U64 i = 0; i < config.bufferSize; ++i) {
                const F64 t = i * dt;
                const F64 expected = config.amplitude * std::sin(2.0 * JST_PI * config.frequency * t);
                REQUIRE_THAT(out.at<F32>(i), Catch::Matchers::WithinAbs(static_cast<F32>(expected), 1e-5f));
            }
        }
    }
}

TEST_CASE("Signal Generator - Cosine F32", "[modules][signal_generator][cosine]") {
    auto implementations = Registry::ListAvailableModules("signal_generator");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            TestContext ctx("signal_generator", impl.device, impl.runtime, impl.provider);

            Modules::SignalGenerator config;
            config.signalType = "cosine";
            config.signalDataType = "F32";
            config.bufferSize = 128;
            config.sampleRate = 1000.0;
            config.frequency = 100.0;
            config.amplitude = 2.0;
            config.phase = 0.0;
            config.dcOffset = 0.5;

            ctx.setConfig(config);

            REQUIRE(ctx.run() == Result::SUCCESS);

            auto& out = ctx.output("signal");
            const F64 dt = 1.0 / config.sampleRate;

            for (U64 i = 0; i < config.bufferSize; ++i) {
                const F64 t = i * dt;
                const F64 expected = config.amplitude * std::cos(2.0 * JST_PI * config.frequency * t) + config.dcOffset;
                REQUIRE_THAT(out.at<F32>(i), Catch::Matchers::WithinAbs(static_cast<F32>(expected), 1e-5f));
            }
        }
    }
}

TEST_CASE("Signal Generator - Chirp CF32 finite output",
          "[modules][signal_generator][chirp][CF32]") {
    auto implementations = Registry::ListAvailableModules("signal_generator");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: "
                        << impl.runtime) {
            TestContext ctx("signal_generator", impl.device, impl.runtime,
                            impl.provider);

            Modules::SignalGenerator config;
            config.signalType = "chirp";
            config.signalDataType = "CF32";
            config.sampleRate = 2048.0;
            config.chirpStartFreq = 50.0;
            config.chirpEndFreq = 400.0;
            config.chirpDuration = 0.25;
            config.bufferSize = 128;
            config.amplitude = 1.0;
            config.phase = 0.1;
            ctx.setConfig(config);

            REQUIRE(ctx.run() == Result::SUCCESS);

            auto& out = ctx.output("signal");
            REQUIRE(out.dtype() == DataType::CF32);
            REQUIRE(out.shape(0) == config.bufferSize);

            for (U64 i = 0; i < config.bufferSize; ++i) {
                const auto sample = out.at<CF32>(i);
                REQUIRE(std::isfinite(sample.real()));
                REQUIRE(std::isfinite(sample.imag()));
            }
        }
    }
}

TEST_CASE("Signal Generator - Sine phase continuity across runs",
          "[modules][signal_generator][state]") {
    auto implementations = Registry::ListAvailableModules("signal_generator");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: "
                        << impl.runtime) {
            TestContext ctx("signal_generator", impl.device, impl.runtime,
                            impl.provider);

            Modules::SignalGenerator config;
            config.signalType = "sine";
            config.signalDataType = "F32";
            config.bufferSize = 32;
            config.sampleRate = 2048.0;
            config.frequency = 128.0;
            config.amplitude = 1.0;
            config.phase = 0.25;
            config.dcOffset = 0.0;
            ctx.setConfig(config);

            REQUIRE(ctx.run() == Result::SUCCESS);
            auto& out = ctx.output("signal");

            const F64 dt = 1.0 / config.sampleRate;
            const F64 t0 = static_cast<F64>(config.bufferSize) * dt;
            const F64 expectedSecondRunFirst = config.amplitude *
                std::sin(2.0 * JST_PI * config.frequency * t0 + config.phase);

            REQUIRE(ctx.run() == Result::SUCCESS);
            REQUIRE_THAT(out.at<F32>(0),
                         Catch::Matchers::WithinAbs(
                             static_cast<F32>(expectedSecondRunFirst), 1e-5f));
        }
    }
}

TEST_CASE("Signal Generator - Validation rejects invalid config",
          "[modules][signal_generator][validation]") {
    auto implementations = Registry::ListAvailableModules("signal_generator");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        SECTION("invalid signalType") {
            TestContext ctx("signal_generator", impl.device, impl.runtime,
                            impl.provider);
            Modules::SignalGenerator config;
            config.signalType = "unknown";
            ctx.setConfig(config);
            REQUIRE(ctx.run() == Result::ERROR);
        }

        SECTION("invalid signalDataType") {
            TestContext ctx("signal_generator", impl.device, impl.runtime,
                            impl.provider);
            Modules::SignalGenerator config;
            config.signalDataType = "I16";
            ctx.setConfig(config);
            REQUIRE(ctx.run() == Result::ERROR);
        }

        SECTION("negative noise variance") {
            TestContext ctx("signal_generator", impl.device, impl.runtime,
                            impl.provider);
            Modules::SignalGenerator config;
            config.noiseVariance = -0.1;
            ctx.setConfig(config);
            REQUIRE(ctx.run() == Result::ERROR);
        }

        SECTION("chirp non-positive duration") {
            TestContext ctx("signal_generator", impl.device, impl.runtime,
                            impl.provider);
            Modules::SignalGenerator config;
            config.signalType = "chirp";
            config.chirpDuration = 0.0;
            ctx.setConfig(config);
            REQUIRE(ctx.run() == Result::ERROR);
        }

        SECTION("chirp negative start frequency") {
            TestContext ctx("signal_generator", impl.device, impl.runtime,
                            impl.provider);
            Modules::SignalGenerator config;
            config.signalType = "chirp";
            config.chirpStartFreq = -1.0;
            ctx.setConfig(config);
            REQUIRE(ctx.run() == Result::ERROR);
        }

        SECTION("chirp negative end frequency") {
            TestContext ctx("signal_generator", impl.device, impl.runtime,
                            impl.provider);
            Modules::SignalGenerator config;
            config.signalType = "chirp";
            config.chirpEndFreq = -1.0;
            ctx.setConfig(config);
            REQUIRE(ctx.run() == Result::ERROR);
        }
    }
}
