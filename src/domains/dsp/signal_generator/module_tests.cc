#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "jetstream/testing.hh"
#include "jetstream/registry.hh"
#include "jetstream/domains/dsp/signal_generator/module.hh"

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

TEST_CASE("Signal Generator - Validation rejects invalid config",
          "[modules][signal_generator][validation]") {
    auto implementations = Registry::ListAvailableModules("signal_generator");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
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

        SECTION("frequency must be finite and non-negative") {
            Modules::SignalGenerator config;
            config.frequency = std::numeric_limits<F64>::infinity();
            RequireSignalGeneratorValidationError(impl, config);

            config = {};
            config.frequency = -1.0;
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
            config.noiseVariance = std::numeric_limits<F64>::infinity();
            RequireSignalGeneratorValidationError(impl, config);

            config = {};
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

        SECTION("chirp start frequency must be finite and non-negative") {
            Modules::SignalGenerator config;
            config.signalType = "chirp";
            config.chirpStartFreq = std::numeric_limits<F64>::infinity();
            RequireSignalGeneratorValidationError(impl, config);

            config = {};
            config.signalType = "chirp";
            config.chirpStartFreq = -1.0;
            RequireSignalGeneratorValidationError(impl, config);
        }

        SECTION("chirp end frequency must be finite and non-negative") {
            Modules::SignalGenerator config;
            config.signalType = "chirp";
            config.chirpEndFreq = std::numeric_limits<F64>::quiet_NaN();
            RequireSignalGeneratorValidationError(impl, config);

            config = {};
            config.signalType = "chirp";
            config.chirpEndFreq = -1.0;
            RequireSignalGeneratorValidationError(impl, config);
        }
    }
}
