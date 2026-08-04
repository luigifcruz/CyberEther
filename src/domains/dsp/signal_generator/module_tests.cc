#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "jetstream/testing.hh"
#include "jetstream/registry.hh"
#include "jetstream/domains/dsp/signal_generator/module.hh"

#include <any>
#include <cmath>
#include <limits>

using namespace Jetstream;

namespace {

void RequireSignalGeneratorValidationError(const Registry::ModuleRegistration& impl,
                                           const Modules::SignalGenerator& config) {
    std::shared_ptr<Module> module;
    REQUIRE(Registry::BuildModule("signal_generator", impl.device, impl.runtime,
                                  impl.provider, module) == Result::SUCCESS);
    REQUIRE(module->create("test", config, {}) == Result::ERROR);
    REQUIRE(module->state() == Module::State::ERRORED);
    REQUIRE(module->outputs().empty());
}

}  // namespace

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
            REQUIRE(out.hasAttribute("sampleAxis"));
            REQUIRE(out.attribute("sampleAxis").type() == typeid(Index));
            REQUIRE(std::any_cast<Index>(out.attribute("sampleAxis")) == Index{0});
            REQUIRE_FALSE(out.hasAttribute("batchAxis"));
            REQUIRE_FALSE(out.hasAttribute("channelAxis"));

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

TEST_CASE("Signal Generator - Signed analytic CF32 sinusoids",
          "[modules][signal_generator][CF32][frequency]") {
    const auto implementations = Registry::ListAvailableModules("signal_generator");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        for (const std::string signalType : {"sine", "cosine"}) {
            for (const F64 frequency : {-500.0, -125.0, 125.0, 500.0}) {
                DYNAMIC_SECTION("Device: " << impl.device << " Runtime: "
                                << impl.runtime << " Type: " << signalType
                                << " Frequency: " << frequency) {
                    Modules::SignalGenerator config;
                    config.signalType = signalType;
                    config.signalDataType = "CF32";
                    config.sampleRate = 1000.0;
                    config.frequency = frequency;
                    config.amplitude = 0.75;
                    config.phase = 0.2;
                    config.dcOffset = 0.1;
                    config.bufferSize = 16;

                    TestContext ctx("signal_generator", impl.device,
                                    impl.runtime, impl.provider);
                    ctx.setConfig(config);
                    REQUIRE(ctx.run() == Result::SUCCESS);
                    const auto& output = ctx.output("signal");
                    REQUIRE(std::any_cast<F32>(
                        output.attribute("frequency")) == 0.0f);
                    REQUIRE(std::any_cast<F32>(
                        output.attribute("sampleRate")) == 1000.0f);

                    for (U64 sample = 0; sample < config.bufferSize; ++sample) {
                        const F64 angle = config.phase + 2.0 * JST_PI *
                            frequency * sample / config.sampleRate;
                        const F64 expectedI = config.amplitude *
                            (signalType == "sine" ? std::sin(angle) :
                                                    std::cos(angle)) +
                            config.dcOffset;
                        const F64 expectedQ = config.amplitude *
                            (signalType == "sine" ? -std::cos(angle) :
                                                    std::sin(angle));
                        REQUIRE_THAT(output.at<CF32>(sample).real(),
                                     Catch::Matchers::WithinAbs(
                                         static_cast<F32>(expectedI), 1e-5f));
                        REQUIRE_THAT(output.at<CF32>(sample).imag(),
                                     Catch::Matchers::WithinAbs(
                                         static_cast<F32>(expectedQ), 1e-5f));
                    }
                }
            }
        }
    }
}

TEST_CASE("Signal Generator - Signed CF32 chirp follows coherent phase",
          "[modules][signal_generator][chirp][CF32][frequency]") {
    const auto implementations = Registry::ListAvailableModules("signal_generator");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: "
                        << impl.runtime) {
            Modules::SignalGenerator config;
            config.signalType = "chirp";
            config.signalDataType = "CF32";
            config.sampleRate = 10e6;
            config.chirpStartFreq = -5e6;
            config.chirpEndFreq = 5e6;
            config.chirpDuration = 1e-6;
            config.amplitude = 1.0;
            config.phase = 0.3;
            config.bufferSize = 12;

            TestContext ctx("signal_generator", impl.device, impl.runtime,
                            impl.provider);
            ctx.setConfig(config);
            REQUIRE(ctx.run() == Result::SUCCESS);
            const auto& output = ctx.output("signal");
            const F64 chirpRate = (config.chirpEndFreq -
                                   config.chirpStartFreq) /
                                  config.chirpDuration;
            const F64 cyclesPerSweep = 0.5 *
                (config.chirpStartFreq + config.chirpEndFreq) *
                config.chirpDuration;
            for (U64 sample = 0; sample < config.bufferSize; ++sample) {
                const F64 time = sample / config.sampleRate;
                const F64 sweep = std::floor(time / config.chirpDuration);
                const F64 localTime = time - sweep * config.chirpDuration;
                const F64 cycles = sweep * cyclesPerSweep +
                    config.chirpStartFreq * localTime +
                    0.5 * chirpRate * localTime * localTime;
                const F64 angle = config.phase + 2.0 * JST_PI * cycles;
                REQUIRE_THAT(output.at<CF32>(sample).real(),
                             Catch::Matchers::WithinAbs(
                                 static_cast<F32>(std::cos(angle)), 1e-4f));
                REQUIRE_THAT(output.at<CF32>(sample).imag(),
                             Catch::Matchers::WithinAbs(
                                 static_cast<F32>(std::sin(angle)), 1e-4f));
            }

            config.chirpStartFreq = 5e6;
            config.chirpEndFreq = -5e6;
            TestContext descendingCtx("signal_generator", impl.device,
                                      impl.runtime, impl.provider);
            descendingCtx.setConfig(config);
            REQUIRE(descendingCtx.run() == Result::SUCCESS);
            const F64 dt = 1.0 / config.sampleRate;
            const F64 descendingRate = (config.chirpEndFreq -
                                        config.chirpStartFreq) /
                                       config.chirpDuration;
            const F64 firstStepCycles = config.chirpStartFreq * dt +
                                        0.5 * descendingRate * dt * dt;
            const F64 firstStepAngle = config.phase +
                                       2.0 * JST_PI * firstStepCycles;
            REQUIRE_THAT(descendingCtx.output("signal").at<CF32>(1).real(),
                         Catch::Matchers::WithinAbs(
                             static_cast<F32>(std::cos(firstStepAngle)), 1e-4f));
            REQUIRE_THAT(descendingCtx.output("signal").at<CF32>(1).imag(),
                         Catch::Matchers::WithinAbs(
                             static_cast<F32>(std::sin(firstStepAngle)), 1e-4f));
        }
    }
}

TEST_CASE("Signal Generator - Repeated chirp preserves carrier phase",
          "[modules][signal_generator][chirp][state]") {
    const auto implementations = Registry::ListAvailableModules("signal_generator");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: "
                        << impl.runtime) {
            Modules::SignalGenerator config;
            config.signalType = "chirp";
            config.signalDataType = "CF32";
            config.sampleRate = 8.0;
            config.chirpStartFreq = 1.0;
            config.chirpEndFreq = 2.0;
            config.chirpDuration = 0.7;
            config.bufferSize = 8;

            TestContext ctx("signal_generator", impl.device, impl.runtime,
                            impl.provider);
            ctx.setConfig(config);
            REQUIRE(ctx.run() == Result::SUCCESS);
            const auto& output = ctx.output("signal");
            const F64 cyclesPerSweep = 0.5 *
                (config.chirpStartFreq + config.chirpEndFreq) *
                config.chirpDuration;
            const F64 boundaryTime = 6.0 / config.sampleRate;
            const F64 localTime = boundaryTime - config.chirpDuration;
            const F64 chirpRate = (config.chirpEndFreq -
                                   config.chirpStartFreq) /
                                  config.chirpDuration;
            const F64 boundaryCycles = cyclesPerSweep +
                config.chirpStartFreq * localTime +
                0.5 * chirpRate * localTime * localTime;
            const F64 boundaryAngle = 2.0 * JST_PI * boundaryCycles;
            REQUIRE_THAT(output.at<CF32>(6).real(),
                         Catch::Matchers::WithinAbs(
                             static_cast<F32>(std::cos(boundaryAngle)), 1e-5f));
            REQUIRE_THAT(output.at<CF32>(6).imag(),
                         Catch::Matchers::WithinAbs(
                             static_cast<F32>(std::sin(boundaryAngle)), 1e-5f));
            REQUIRE(std::abs(output.at<CF32>(6) - CF32{1.0f, 0.0f}) > 0.1f);
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

TEST_CASE("Signal Generator - Zero noise variance produces DC offset",
          "[modules][signal_generator][noise]") {
    auto implementations = Registry::ListAvailableModules("signal_generator");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: "
                        << impl.runtime) {
            TestContext ctx("signal_generator", impl.device, impl.runtime,
                            impl.provider);

            Modules::SignalGenerator config;
            config.signalType = "noise";
            config.signalDataType = "F32";
            config.bufferSize = 32;
            config.amplitude = 2.0;
            config.dcOffset = 0.25;
            config.noiseVariance = 0.0;
            ctx.setConfig(config);

            REQUIRE(ctx.run() == Result::SUCCESS);
            auto& out = ctx.output("signal");
            for (U64 i = 0; i < config.bufferSize; ++i) {
                REQUIRE_THAT(out.at<F32>(i),
                             Catch::Matchers::WithinAbs(0.25f, 1e-6f));
            }
        }
    }
}

TEST_CASE("Signal Generator - Noise output remains finite at extreme scale",
          "[modules][signal_generator][noise][range]") {
    const auto implementations = Registry::ListAvailableModules("signal_generator");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: "
                        << impl.runtime) {
            Modules::SignalGenerator config;
            config.signalType = "noise";
            config.signalDataType = "CF32";
            config.amplitude = std::numeric_limits<F32>::max();
            config.noiseVariance = std::numeric_limits<F32>::max();
            config.bufferSize = 32;

            TestContext ctx("signal_generator", impl.device, impl.runtime,
                            impl.provider);
            ctx.setConfig(config);
            REQUIRE(ctx.run() == Result::SUCCESS);
            const auto& output = ctx.output("signal");
            for (U64 sample = 0; sample < config.bufferSize; ++sample) {
                REQUIRE(std::isfinite(output.at<CF32>(sample).real()));
                REQUIRE(std::isfinite(output.at<CF32>(sample).imag()));
            }
        }
    }
}

TEST_CASE("Signal Generator - Negative phase remains normalized",
          "[modules][signal_generator][phase]") {
    auto implementations = Registry::ListAvailableModules("signal_generator");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        for (const std::string signalType : {"square", "sawtooth", "triangle"}) {
            DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime
                            << " Type: " << signalType) {
                TestContext ctx("signal_generator", impl.device, impl.runtime,
                                impl.provider);

                Modules::SignalGenerator config;
                config.signalType = signalType;
                config.signalDataType = "F32";
                config.bufferSize = 16;
                config.sampleRate = 1024.0;
                config.frequency = 0.0;
                config.amplitude = 1.0;
                config.phase = -JST_PI;
                config.dcOffset = 0.0;
                ctx.setConfig(config);

                REQUIRE(ctx.run() == Result::SUCCESS);
                auto& out = ctx.output("signal");
                for (U64 i = 0; i < config.bufferSize; ++i) {
                    REQUIRE(std::isfinite(out.at<F32>(i)));
                    REQUIRE(std::abs(out.at<F32>(i)) <= 1.0f);
                }
            }
        }
    }
}

TEST_CASE("Signal Generator - Sine phase continuity across computes",
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
            config.bufferSize = 31;
            config.sampleRate = 2048.0;
            config.frequency = 128.0;
            config.amplitude = 1.0;
            config.phase = 0.25;
            config.dcOffset = 0.0;
            ctx.setConfig(config);

            REQUIRE(ctx.start() == Result::SUCCESS);
            REQUIRE(ctx.compute() == Result::SUCCESS);
            auto& out = ctx.output("signal");
            const F32 firstRunFirst = out.at<F32>(0);

            const F64 dt = 1.0 / config.sampleRate;
            const F64 t0 = static_cast<F64>(config.bufferSize) * dt;
            const F64 expectedSecondRunFirst = config.amplitude *
                std::sin(2.0 * JST_PI * config.frequency * t0 + config.phase);

            REQUIRE(ctx.compute() == Result::SUCCESS);
            const F32 secondRunFirst = out.at<F32>(0);
            REQUIRE_THAT(secondRunFirst,
                         Catch::Matchers::WithinAbs(
                             static_cast<F32>(expectedSecondRunFirst), 1e-5f));
            REQUIRE(std::abs(secondRunFirst - firstRunFirst) > 1e-5f);

            config.bufferSize = 17;
            REQUIRE(ctx.reconfigure(config) == Result::RECREATE);
            REQUIRE(ctx.stop() == Result::SUCCESS);
            REQUIRE(ctx.start() == Result::SUCCESS);
            REQUIRE(ctx.compute() == Result::SUCCESS);
            REQUIRE(ctx.output("signal").shape(0) == 17);
            REQUIRE(ctx.stop() == Result::SUCCESS);
        }
    }
}

TEST_CASE("Signal Generator - Scalar reconfiguration preserves phase",
          "[modules][signal_generator][state][reconfigure]") {
    const auto implementations = Registry::ListAvailableModules("signal_generator");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: "
                        << impl.runtime) {
            Modules::SignalGenerator config;
            config.signalType = "sine";
            config.signalDataType = "F32";
            config.sampleRate = 1000.0;
            config.frequency = 125.0;
            config.amplitude = 1.0;
            config.phase = 0.1;
            config.bufferSize = 5;

            TestContext ctx("signal_generator", impl.device, impl.runtime,
                            impl.provider);
            ctx.setConfig(config);
            REQUIRE(ctx.start() == Result::SUCCESS);
            REQUIRE(ctx.compute() == Result::SUCCESS);
            const U64 outputId = ctx.output("signal").id();

            const F64 phaseBeforeUpdate = config.phase + 2.0 * JST_PI *
                config.frequency * config.bufferSize / config.sampleRate;
            config.frequency = 250.0;
            config.amplitude = 2.0;
            config.phase += 0.25;
            REQUIRE(ctx.reconfigure(config) == Result::SUCCESS);
            REQUIRE(ctx.output("signal").id() == outputId);
            REQUIRE(std::any_cast<F32>(
                ctx.output("signal").attribute("sampleRate")) == 1000.0f);

            REQUIRE(ctx.compute() == Result::SUCCESS);
            const F32 expected = static_cast<F32>(
                config.amplitude * std::sin(phaseBeforeUpdate + 0.25));
            REQUIRE_THAT(ctx.output("signal").at<F32>(0),
                         Catch::Matchers::WithinAbs(expected, 1e-5f));

            config.phase = std::numeric_limits<F64>::max();
            REQUIRE(ctx.reconfigure(config) == Result::SUCCESS);
            config.phase = -std::numeric_limits<F64>::max();
            REQUIRE(ctx.reconfigure(config) == Result::SUCCESS);
            REQUIRE(ctx.compute() == Result::SUCCESS);
            REQUIRE(std::isfinite(ctx.output("signal").at<F32>(0)));

            config.sampleRate = 2000.0;
            REQUIRE(ctx.reconfigure(config) == Result::RECREATE);
            REQUIRE(ctx.stop() == Result::SUCCESS);
        }
    }
}

TEST_CASE("Signal Generator - Chirp reconfiguration preserves phase and position",
          "[modules][signal_generator][chirp][state][reconfigure]") {
    const auto implementations = Registry::ListAvailableModules("signal_generator");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: "
                        << impl.runtime) {
            Modules::SignalGenerator config;
            config.signalType = "chirp";
            config.signalDataType = "CF32";
            config.sampleRate = 10.0;
            config.chirpStartFreq = 1.0;
            config.chirpEndFreq = 2.0;
            config.chirpDuration = 1.0;
            config.bufferSize = 3;

            TestContext ctx("signal_generator", impl.device, impl.runtime,
                            impl.provider);
            ctx.setConfig(config);
            REQUIRE(ctx.start() == Result::SUCCESS);
            REQUIRE(ctx.compute() == Result::SUCCESS);

            config.chirpDuration = 2.0;
            REQUIRE(ctx.reconfigure(config) == Result::SUCCESS);
            REQUIRE(ctx.compute() == Result::SUCCESS);
            const auto& output = ctx.output("signal");

            const F64 cyclesBeforeUpdate = 0.3 + 0.5 * 0.3 * 0.3;
            const F64 firstAngle = 2.0 * JST_PI * cyclesBeforeUpdate;
            REQUIRE_THAT(output.at<CF32>(0).real(),
                         Catch::Matchers::WithinAbs(
                             static_cast<F32>(std::cos(firstAngle)), 1e-5f));
            REQUIRE_THAT(output.at<CF32>(0).imag(),
                         Catch::Matchers::WithinAbs(
                             static_cast<F32>(std::sin(firstAngle)), 1e-5f));

            const F64 remappedTime = 0.6;
            const F64 newChirpRate = 0.5;
            const F64 nextCycles = cyclesBeforeUpdate +
                (config.chirpStartFreq + newChirpRate * remappedTime) * 0.1 +
                0.5 * newChirpRate * 0.1 * 0.1;
            const F64 nextAngle = 2.0 * JST_PI * nextCycles;
            REQUIRE_THAT(output.at<CF32>(1).real(),
                         Catch::Matchers::WithinAbs(
                             static_cast<F32>(std::cos(nextAngle)), 1e-5f));
            REQUIRE_THAT(output.at<CF32>(1).imag(),
                         Catch::Matchers::WithinAbs(
                             static_cast<F32>(std::sin(nextAngle)), 1e-5f));
            REQUIRE(ctx.stop() == Result::SUCCESS);
        }
    }
}

TEST_CASE("Signal Generator - Inactive fields do not invalidate waveform",
          "[modules][signal_generator][validation][inactive]") {
    const auto implementations = Registry::ListAvailableModules("signal_generator");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: "
                        << impl.runtime) {
            Modules::SignalGenerator config;
            config.signalType = "noise";
            config.frequency = std::numeric_limits<F64>::quiet_NaN();
            config.phase = std::numeric_limits<F64>::infinity();
            config.chirpStartFreq = std::numeric_limits<F64>::infinity();
            config.bufferSize = 8;

            TestContext ctx("signal_generator", impl.device, impl.runtime,
                            impl.provider);
            ctx.setConfig(config);
            REQUIRE(ctx.run() == Result::SUCCESS);

            config = {};
            config.signalType = "cosine";
            config.noiseVariance = std::numeric_limits<F64>::infinity();
            TestContext periodicCtx("signal_generator", impl.device,
                                    impl.runtime, impl.provider);
            periodicCtx.setConfig(config);
            REQUIRE(periodicCtx.run() == Result::SUCCESS);
        }
    }
}

TEST_CASE("Signal Generator - Validation rejects invalid config",
          "[modules][signal_generator][validation]") {
    auto implementations = Registry::ListAvailableModules("signal_generator");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            SECTION("invalid signalType") {
                Modules::SignalGenerator config;
                config.signalType = "unknown";
                RequireSignalGeneratorValidationError(impl, config);
            }

            SECTION("invalid signalDataType") {
                Modules::SignalGenerator config;
                config.signalDataType = "I16";
                RequireSignalGeneratorValidationError(impl, config);
            }

            SECTION("sample rate must be finite and positive") {
                Modules::SignalGenerator config;
                config.sampleRate = std::numeric_limits<F64>::quiet_NaN();
                RequireSignalGeneratorValidationError(impl, config);

                config = {};
                config.sampleRate = 0.0;
                RequireSignalGeneratorValidationError(impl, config);

                config = {};
                config.sampleRate = std::numeric_limits<F64>::max();
                RequireSignalGeneratorValidationError(impl, config);

                config = {};
                config.sampleRate = std::numeric_limits<F64>::denorm_min();
                RequireSignalGeneratorValidationError(impl, config);
            }

            SECTION("frequency must be finite and within the type's Nyquist range") {
                Modules::SignalGenerator config;
                config.frequency = std::numeric_limits<F64>::infinity();
                RequireSignalGeneratorValidationError(impl, config);

                config = {};
                config.frequency = -1.0;
                RequireSignalGeneratorValidationError(impl, config);

                config = {};
                config.sampleRate = 1000.0;
                config.frequency = 501.0;
                RequireSignalGeneratorValidationError(impl, config);

                config = {};
                config.signalDataType = "CF32";
                config.sampleRate = 1000.0;
                config.frequency = -501.0;
                RequireSignalGeneratorValidationError(impl, config);
            }

            SECTION("amplitude must be finite and non-negative") {
                Modules::SignalGenerator config;
                config.amplitude = std::numeric_limits<F64>::quiet_NaN();
                RequireSignalGeneratorValidationError(impl, config);

                config = {};
                config.amplitude = -1.0;
                RequireSignalGeneratorValidationError(impl, config);

                config = {};
                config.amplitude = std::numeric_limits<F64>::max();
                RequireSignalGeneratorValidationError(impl, config);
            }

            SECTION("phase must be finite") {
                Modules::SignalGenerator config;
                config.phase = std::numeric_limits<F64>::infinity();
                RequireSignalGeneratorValidationError(impl, config);
            }

            SECTION("DC offset must be finite") {
                Modules::SignalGenerator config;
                config.dcOffset = std::numeric_limits<F64>::quiet_NaN();
                RequireSignalGeneratorValidationError(impl, config);

                config = {};
                config.amplitude = std::numeric_limits<F32>::max();
                config.dcOffset = std::numeric_limits<F32>::max();
                RequireSignalGeneratorValidationError(impl, config);
            }

            SECTION("buffer size must be nonzero and representable") {
                Modules::SignalGenerator config;
                config.bufferSize = 0;
                RequireSignalGeneratorValidationError(impl, config);

                config = {};
                config.bufferSize = std::numeric_limits<U64>::max();
                RequireSignalGeneratorValidationError(impl, config);

                if (impl.device == DeviceType::CPU) {
                    config = {};
                    config.bufferSize = std::numeric_limits<U64>::max() /
                                        DataTypeSize(DataType::F32);
                    RequireSignalGeneratorValidationError(impl, config);
                }
            }

            SECTION("noise variance must be finite and non-negative") {
                Modules::SignalGenerator config;
                config.signalType = "noise";
                config.noiseVariance = std::numeric_limits<F64>::infinity();
                RequireSignalGeneratorValidationError(impl, config);

                config = {};
                config.signalType = "noise";
                config.noiseVariance = -0.1;
                RequireSignalGeneratorValidationError(impl, config);
            }

            SECTION("chirp duration must be finite and positive") {
                Modules::SignalGenerator config;
                config.signalType = "chirp";
                config.chirpDuration = std::numeric_limits<F64>::quiet_NaN();
                RequireSignalGeneratorValidationError(impl, config);

                config = {};
                config.signalType = "chirp";
                config.chirpDuration = 0.0;
                RequireSignalGeneratorValidationError(impl, config);

                config = {};
                config.signalType = "chirp";
                config.chirpDuration = std::numeric_limits<F64>::denorm_min();
                RequireSignalGeneratorValidationError(impl, config);
            }

            SECTION("chirp start frequency must be in the Nyquist range") {
                Modules::SignalGenerator config;
                config.signalType = "chirp";
                config.chirpStartFreq = std::numeric_limits<F64>::infinity();
                RequireSignalGeneratorValidationError(impl, config);

                config = {};
                config.signalType = "chirp";
                config.chirpStartFreq = -1.0;
                RequireSignalGeneratorValidationError(impl, config);

                config = {};
                config.signalType = "chirp";
                config.sampleRate = 1000.0;
                config.chirpStartFreq = 501.0;
                RequireSignalGeneratorValidationError(impl, config);
            }

            SECTION("chirp end frequency must be in the Nyquist range") {
                Modules::SignalGenerator config;
                config.signalType = "chirp";
                config.chirpEndFreq = std::numeric_limits<F64>::quiet_NaN();
                RequireSignalGeneratorValidationError(impl, config);

                config = {};
                config.signalType = "chirp";
                config.chirpEndFreq = -1.0;
                RequireSignalGeneratorValidationError(impl, config);

                config = {};
                config.signalType = "chirp";
                config.signalDataType = "CF32";
                config.sampleRate = 1000.0;
                config.chirpEndFreq = -501.0;
                RequireSignalGeneratorValidationError(impl, config);
            }
        }
    }
}
