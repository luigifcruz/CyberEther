#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "jetstream/testing.hh"
#include "jetstream/registry.hh"
#include "jetstream/domains/dsp/fm/module.hh"

#include <any>
#include <cmath>
#include <limits>
#include <unordered_set>

using namespace Jetstream;

namespace {

void RequireFmValidationError(const Registry::ModuleRegistration& impl,
                              const Modules::FM& config,
                              const DataType dtype) {
    Tensor input;
    REQUIRE(input.create(impl.device, dtype, {16}) == Result::SUCCESS);
    REQUIRE(input.setAttribute("sampleAxis", Index{0}) == Result::SUCCESS);

    TensorMap inputs;
    inputs["signal"].requested("test", "signal");
    inputs["signal"].tensor = input;

    std::shared_ptr<Module> module;
    REQUIRE(Registry::BuildModule("fm", impl.device, impl.runtime,
                                  impl.provider, module) == Result::SUCCESS);
    REQUIRE(module->create("test", config, inputs) == Result::ERROR);
    REQUIRE(module->state() == Module::State::ERRORED);
    REQUIRE(module->outputs().empty());
}

void RequireFmSignalValidationError(const Registry::ModuleRegistration& impl,
                                    const Tensor& input,
                                    const Modules::FM& config = Modules::FM{}) {
    TensorMap inputs;
    inputs["signal"].requested("test", "signal");
    inputs["signal"].tensor = input;

    std::shared_ptr<Module> module;
    REQUIRE(Registry::BuildModule("fm", impl.device, impl.runtime,
                                  impl.provider, module) == Result::SUCCESS);
    REQUIRE(module->create("test", config, inputs) == Result::ERROR);
    REQUIRE(module->state() == Module::State::ERRORED);
    REQUIRE(module->outputs().empty());
}

F32 ToneAmplitude(const Tensor& signal, U64 channel, F32 frequency,
                  F32 sampleRate, U64 firstSample) {
    F64 inPhase = 0.0;
    F64 quadrature = 0.0;
    const U64 sampleCount = signal.shape(0) - firstSample;
    for (U64 sample = firstSample; sample < signal.shape(0); ++sample) {
        const F64 phase = 2.0 * JST_PI * static_cast<F64>(frequency) *
                          static_cast<F64>(sample) / sampleRate;
        const F64 value = signal.at<F32>(sample, channel);
        inPhase += value * std::sin(phase);
        quadrature += value * std::cos(phase);
    }
    return static_cast<F32>(2.0 * std::hypot(inPhase, quadrature) /
                            static_cast<F64>(sampleCount));
}

}  // namespace

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
            REQUIRE(input.setAttribute("sampleAxis", Index{0}) == Result::SUCCESS);

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

TEST_CASE("FM - Validation Rejects Invalid Configuration Before Create",
          "[modules][fm][validation]") {
    auto implementations = Registry::ListAvailableModules("fm");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            SECTION("sample rate must be finite and in the supported range") {
                Modules::FM config;
                config.sampleRate = 0.0f;
                RequireFmValidationError(impl, config, DataType::CF32);

                config.sampleRate = std::numeric_limits<F32>::quiet_NaN();
                RequireFmValidationError(impl, config, DataType::CF32);

                config.sampleRate = 25e6f;
                RequireFmValidationError(impl, config, DataType::CF32);
            }

            SECTION("mode must be narrow or wide") {
                Modules::FM config;
                config.mode = "invalid";
                RequireFmValidationError(impl, config, DataType::CF32);
            }

            SECTION("de-emphasis must be none, 50us, or 75us") {
                Modules::FM config;
                config.deemphasis = "invalid";
                RequireFmValidationError(impl, config, DataType::CF32);
            }

            SECTION("wideband sample rate must prevent discriminator wrapping") {
                Modules::FM config;
                config.mode = "wide";
                config.sampleRate = 120e3f;
                RequireFmValidationError(impl, config, DataType::CF32);

                config.sampleRate = 199e3f;
                RequireFmValidationError(impl, config, DataType::CF32);
            }

            SECTION("native CPU input must be CF32") {
                Modules::FM config;
                RequireFmValidationError(impl, config, DataType::F32);
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
            const F32 phaseIncrement = 2.0f * static_cast<F32>(JST_PI) * freqOffset / sampleRate;

            Tensor input;
            REQUIRE(input.create(DeviceType::CPU, DataType::CF32, {bufferSize}) == Result::SUCCESS);
            REQUIRE(input.setAttribute("sampleAxis", Index{0}) == Result::SUCCESS);

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
            const F32 ref = 1.0f / (2.0f * static_cast<F32>(JST_PI) * kf);
            const F32 expected = phaseIncrement * ref;

            for (U64 i = 1; i < bufferSize; ++i) {
                REQUIRE_THAT(out.at<F32>(i), Catch::Matchers::WithinAbs(expected, 0.01f));
            }
        }
    }
}

TEST_CASE("FM - Narrowband Applies Selected De-emphasis",
          "[modules][fm][deemphasis]") {
    const auto implementations = Registry::ListAvailableModules("fm");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            constexpr F32 sampleRate = 240e3f;
            constexpr F32 frequencyOffset = 10e3f;
            const F32 phaseIncrement = 2.0f * static_cast<F32>(JST_PI) *
                                       frequencyOffset / sampleRate;

            Tensor input;
            REQUIRE(input.create(DeviceType::CPU, DataType::CF32, {8}) ==
                    Result::SUCCESS);
            REQUIRE(input.setAttribute("sampleAxis", Index{0}) == Result::SUCCESS);
            for (U64 sample = 0; sample < input.size(); ++sample) {
                input.at<CF32>(sample) = std::polar(
                    1.0f, static_cast<F32>(sample) * phaseIncrement);
            }

            TestContext ctx("fm", impl.device, impl.runtime, impl.provider);
            Modules::FM config;
            config.deemphasis = "50us";
            config.sampleRate = sampleRate;
            ctx.setConfig(config);
            ctx.setInput("signal", input);

            REQUIRE(ctx.run() == Result::SUCCESS);
            const auto& output = ctx.output("signal");
            const F32 ref = 1.0f /
                (2.0f * static_cast<F32>(JST_PI) * (100e3f / sampleRate));
            const F32 raw = phaseIncrement * ref;
            const F32 alpha = 1.0f - std::exp(-1.0f /
                                              (sampleRate * 50e-6f));
            REQUIRE_THAT(output.at<F32>(1),
                         Catch::Matchers::WithinAbs(alpha * raw, 1e-5f));
            REQUIRE_THAT(output.at<F32>(2),
                         Catch::Matchers::WithinAbs(
                             (1.0f - (1.0f - alpha) * (1.0f - alpha)) * raw,
                             1e-5f));
        }
    }
}

TEST_CASE("FM - Wideband Decodes Stereo Multiplex",
          "[modules][fm][stereo]") {
    const auto implementations = Registry::ListAvailableModules("fm");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            constexpr F32 sampleRate = 200e3f;
            constexpr U64 bufferSize = 8192;
            constexpr F32 left = 0.4f;
            constexpr F32 right = -0.2f;
            constexpr F32 pilotOffset = 0.37f;
            const F32 pilotIncrement = 2.0f * static_cast<F32>(JST_PI) *
                                       19e3f / sampleRate;
            const F32 modulationScale = 2.0f * static_cast<F32>(JST_PI) *
                                        75e3f / sampleRate;

            Tensor input;
            REQUIRE(input.create(DeviceType::CPU, DataType::CF32, {bufferSize}) ==
                    Result::SUCCESS);
            REQUIRE(input.setAttribute("sampleAxis", Index{0}) == Result::SUCCESS);

            F32 phase = 0.0f;
            for (U64 sample = 0; sample < bufferSize; ++sample) {
                input.at<CF32>(sample) = std::polar(1.0f, phase);
                const F32 pilotPhase = static_cast<F32>(sample) *
                                       pilotIncrement + pilotOffset;
                const F32 sum = 0.5f * (left + right);
                const F32 difference = 0.5f * (left - right);
                const F32 multiplex = 0.9f *
                    (sum + difference * std::sin(2.0f * pilotPhase)) +
                    0.1f * std::sin(pilotPhase);
                phase += modulationScale * multiplex;
            }

            TestContext ctx("fm", impl.device, impl.runtime, impl.provider);
            Modules::FM config;
            config.mode = "wide";
            config.sampleRate = sampleRate;
            ctx.setConfig(config);
            ctx.setInput("signal", input);

            REQUIRE(ctx.run() == Result::SUCCESS);
            const auto& output = ctx.output("signal");
            REQUIRE(output.shape() == Shape{bufferSize, 2});
            REQUIRE(std::any_cast<Index>(output.attribute("sampleAxis")) == Index{0});
            REQUIRE(std::any_cast<Index>(output.attribute("channelAxis")) == Index{1});

            F32 leftAverage = 0.0f;
            F32 rightAverage = 0.0f;
            constexpr U64 averageSize = 2048;
            for (U64 sample = bufferSize - averageSize;
                 sample < bufferSize; ++sample) {
                leftAverage += output.at<F32>(sample, 0);
                rightAverage += output.at<F32>(sample, 1);
            }
            leftAverage /= static_cast<F32>(averageSize);
            rightAverage /= static_cast<F32>(averageSize);
            REQUIRE_THAT(leftAverage,
                         Catch::Matchers::WithinAbs(0.9f * left, 0.01f));
            REQUIRE_THAT(rightAverage,
                         Catch::Matchers::WithinAbs(0.9f * right, 0.01f));
        }
    }
}

TEST_CASE("FM - Wideband Separates Audio Tones and Rejects Pilot",
          "[modules][fm][stereo][spectral]") {
    const auto implementations = Registry::ListAvailableModules("fm");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            constexpr F32 sampleRate = 240e3f;
            constexpr U64 bufferSize = 24000;
            constexpr F32 leftFrequency = 15e3f;
            constexpr F32 rightFrequency = 1e3f;
            constexpr F32 pilotFrequency = 19e3f;
            constexpr F32 pilotOffset = 0.41f;
            const F64 modulationScale = 2.0 * JST_PI * 75e3 / sampleRate;

            Tensor input;
            REQUIRE(input.create(DeviceType::CPU, DataType::CF32,
                                 {bufferSize}) == Result::SUCCESS);
            REQUIRE(input.setAttribute("sampleAxis", Index{0}) == Result::SUCCESS);

            F64 carrierPhase = 0.0;
            for (U64 sample = 0; sample < bufferSize; ++sample) {
                input.at<CF32>(sample) = std::polar(
                    1.0f, static_cast<F32>(carrierPhase));
                const F64 time = static_cast<F64>(sample) / sampleRate;
                const F64 left = std::sin(2.0 * JST_PI *
                                          leftFrequency * time);
                const F64 right = std::sin(2.0 * JST_PI *
                                           rightFrequency * time);
                const F64 pilotPhase = 2.0 * JST_PI * pilotFrequency * time +
                                       pilotOffset;
                const F64 multiplex = 0.9 *
                    (0.5 * (left + right) +
                     0.5 * (left - right) * std::sin(2.0 * pilotPhase)) +
                    0.1 * std::sin(pilotPhase);
                carrierPhase += modulationScale * multiplex;
            }

            TestContext ctx("fm", impl.device, impl.runtime, impl.provider);
            Modules::FM config;
            config.mode = "wide";
            config.sampleRate = sampleRate;
            ctx.setConfig(config);
            ctx.setInput("signal", input);

            REQUIRE(ctx.run() == Result::SUCCESS);
            const auto& output = ctx.output("signal");
            constexpr U64 firstSample = bufferSize / 2;
            REQUIRE(ToneAmplitude(output, 0, leftFrequency, sampleRate,
                                  firstSample) > 0.5f);
            REQUIRE(ToneAmplitude(output, 1, rightFrequency, sampleRate,
                                  firstSample) > 0.75f);
            REQUIRE(ToneAmplitude(output, 0, rightFrequency, sampleRate,
                                  firstSample) < 0.05f);
            REQUIRE(ToneAmplitude(output, 1, leftFrequency, sampleRate,
                                  firstSample) < 0.05f);
            REQUIRE(ToneAmplitude(output, 0, pilotFrequency, sampleRate,
                                  firstSample) < 0.01f);
            REQUIRE(ToneAmplitude(output, 1, pilotFrequency, sampleRate,
                                  firstSample) < 0.01f);
        }
    }
}

TEST_CASE("FM - Non-finite Input Does Not Poison Recursive State",
          "[modules][fm][state][nonfinite]") {
    const auto implementations = Registry::ListAvailableModules("fm");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        for (const char* deemphasis : {"none", "50us"}) {
            DYNAMIC_SECTION("Device: " << impl.device << " Runtime: "
                            << impl.runtime << " De-emphasis: " << deemphasis) {
                constexpr F32 sampleRate = 240e3f;
                constexpr F32 phaseIncrement = 0.2f;
                Tensor input;
                REQUIRE(input.create(DeviceType::CPU, DataType::CF32, {6}) ==
                        Result::SUCCESS);
                REQUIRE(input.setAttribute("sampleAxis", Index{0}) ==
                        Result::SUCCESS);
                for (U64 sample = 0; sample < input.size(); ++sample) {
                    input.at<CF32>(sample) = std::polar(
                        1.0f, static_cast<F32>(sample) * phaseIncrement);
                }
                const F32 nan = std::numeric_limits<F32>::quiet_NaN();
                input.at<CF32>(2) = CF32{nan, nan};

                TestContext ctx("fm", impl.device, impl.runtime, impl.provider);
                Modules::FM config;
                config.deemphasis = deemphasis;
                config.sampleRate = sampleRate;
                ctx.setConfig(config);
                ctx.setInput("signal", input);

                REQUIRE(ctx.run() == Result::SUCCESS);
                const auto& output = ctx.output("signal");
                REQUIRE_FALSE(std::isfinite(output.at<F32>(2)));
                REQUIRE_FALSE(std::isfinite(output.at<F32>(3)));
                REQUIRE(std::isfinite(output.at<F32>(4)));
                REQUIRE(std::isfinite(output.at<F32>(5)));
            }
        }
    }
}

TEST_CASE("FM - Preserves phase differences across submissions",
          "[modules][fm][phase][state]") {
    const auto implementations = Registry::ListAvailableModules("fm");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        if (impl.device != DeviceType::CPU) {
            continue;
        }

        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            constexpr F32 sampleRate = 240e3f;
            constexpr F32 frequencyOffset = 10e3f;
            const F32 phaseIncrement = 2.0f * static_cast<F32>(JST_PI) *
                                       frequencyOffset / sampleRate;

            Tensor input;
            REQUIRE(input.create(DeviceType::CPU, DataType::CF32, {4}) ==
                    Result::SUCCESS);
            REQUIRE(input.setAttribute("sampleAxis", Index{0}) == Result::SUCCESS);
            for (U64 i = 0; i < input.size(); ++i) {
                const F32 phase = static_cast<F32>(i) * phaseIncrement;
                input.at<CF32>(i) = std::polar(1.0f, phase);
            }

            TensorMap inputs;
            inputs["signal"].requested("test", "signal");
            inputs["signal"].tensor = input;

            Modules::FM config;
            config.sampleRate = sampleRate;
            std::shared_ptr<Module> module;
            REQUIRE(Registry::BuildModule("fm", impl.device, impl.runtime,
                                          impl.provider, module) == Result::SUCCESS);
            REQUIRE(module->create("test", config, inputs) == Result::SUCCESS);

            Runtime runtime("test", impl.device, impl.runtime);
            REQUIRE(runtime.create({{"test", module}}) == Result::SUCCESS);
            std::unordered_set<std::string> skippedModules;
            std::unordered_set<std::string> failedModules;
            REQUIRE(runtime.compute({}, skippedModules, failedModules) ==
                    Result::SUCCESS);

            for (U64 i = 0; i < input.size(); ++i) {
                const F32 phase = static_cast<F32>(i + input.size()) *
                                  phaseIncrement;
                input.at<CF32>(i) = std::polar(1.0f, phase);
            }
            skippedModules.clear();
            failedModules.clear();
            REQUIRE(runtime.compute({}, skippedModules, failedModules) ==
                    Result::SUCCESS);

            const F32 kf = 100e3f / sampleRate;
            const F32 ref = 1.0f / (2.0f * static_cast<F32>(JST_PI) * kf);
            const Tensor output = module->outputs().at("signal").tensor;
            REQUIRE_THAT(output.at<F32>(0),
                         Catch::Matchers::WithinAbs(phaseIncrement * ref, 0.01f));

            REQUIRE(runtime.destroy() == Result::SUCCESS);
            REQUIRE(module->destroy() == Result::SUCCESS);
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
            REQUIRE(input.setAttribute("sampleAxis", Index{0}) == Result::SUCCESS);

            for (U64 i = 0; i < bufferSize; ++i) {
                input.at<CF32>(i) = CF32(1.0f, 0.0f);
            }

            ctx.setInput("signal", input);

            REQUIRE(ctx.run() == Result::SUCCESS);

            auto& out = ctx.output("signal");

            REQUIRE(out.size() == bufferSize);
            REQUIRE(std::any_cast<Index>(out.attribute("sampleAxis")) == Index{0});
        }
    }
}

TEST_CASE("FM - Keeps channels independent for either sample-axis placement",
          "[modules][fm][axes]") {
    const auto implementations = Registry::ListAvailableModules("fm");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        for (const bool sampleFirst : {false, true}) {
            DYNAMIC_SECTION("Device: " << impl.device << " Runtime: "
                            << impl.runtime << " sampleFirst: " << sampleFirst) {
                constexpr F32 sampleRate = 100.0e3f;
                TestContext ctx("fm", impl.device, impl.runtime, impl.provider);
                Modules::FM config;
                config.sampleRate = sampleRate;
                ctx.setConfig(config);

                Tensor input;
                REQUIRE(input.create(DeviceType::CPU, DataType::CF32,
                                     sampleFirst ? Shape{4, 2} : Shape{2, 4}) ==
                        Result::SUCCESS);
                const Index sampleAxis = sampleFirst ? 0 : 1;
                const Index channelAxis = sampleFirst ? 1 : 0;
                REQUIRE(input.setAttribute("sampleAxis", sampleAxis) ==
                        Result::SUCCESS);
                REQUIRE(input.setAttribute("channelAxis", channelAxis) ==
                        Result::SUCCESS);

                for (U64 channel = 0; channel < 2; ++channel) {
                    const F32 start = channel == 0 ? 0.0f : 1.0f;
                    const F32 increment = channel == 0 ? 0.4f : -0.3f;
                    for (U64 sample = 0; sample < 4; ++sample) {
                        const CF32 value = std::polar(
                            1.0f, start + static_cast<F32>(sample) * increment);
                        if (sampleFirst) {
                            input.at<CF32>(sample, channel) = value;
                        } else {
                            input.at<CF32>(channel, sample) = value;
                        }
                    }
                }
                ctx.setInput("signal", input);

                REQUIRE(ctx.run() == Result::SUCCESS);
                const auto& output = ctx.output("signal");
                const F32 ref = 1.0f / (2.0f * static_cast<F32>(JST_PI));
                for (U64 channel = 0; channel < 2; ++channel) {
                    const F32 increment = channel == 0 ? 0.4f : -0.3f;
                    for (U64 sample = 0; sample < 4; ++sample) {
                        const F32 value = sampleFirst
                            ? output.at<F32>(sample, channel)
                            : output.at<F32>(channel, sample);
                        REQUIRE_THAT(value, Catch::Matchers::WithinAbs(
                            sample == 0 ? 0.0f : increment * ref, 1.0e-5f));
                    }
                }
                REQUIRE(std::any_cast<Index>(output.attribute("sampleAxis")) ==
                        sampleAxis);
                REQUIRE(std::any_cast<Index>(output.attribute("channelAxis")) ==
                        channelAxis);
            }
        }
    }
}

TEST_CASE("FM - Sequences batches independently per channel",
          "[modules][fm][axes][batch]") {
    const auto implementations = Registry::ListAvailableModules("fm");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            constexpr F32 sampleRate = 100.0e3f;
            TestContext ctx("fm", impl.device, impl.runtime, impl.provider);
            Modules::FM config;
            config.sampleRate = sampleRate;
            ctx.setConfig(config);

            Tensor input;
            REQUIRE(input.create(DeviceType::CPU, DataType::CF32, {2, 2, 2}) ==
                    Result::SUCCESS);
            REQUIRE(input.setAttribute("batchAxis", Index{0}) == Result::SUCCESS);
            REQUIRE(input.setAttribute("channelAxis", Index{1}) == Result::SUCCESS);
            REQUIRE(input.setAttribute("sampleAxis", Index{2}) == Result::SUCCESS);
            for (U64 batch = 0; batch < 2; ++batch) {
                for (U64 channel = 0; channel < 2; ++channel) {
                    const F32 increment = channel == 0 ? 0.2f : -0.35f;
                    for (U64 sample = 0; sample < 2; ++sample) {
                        const U64 sequenceSample = batch * 2 + sample;
                        input.at<CF32>(batch, channel, sample) = std::polar(
                            1.0f, static_cast<F32>(sequenceSample) * increment);
                    }
                }
            }
            ctx.setInput("signal", input);

            REQUIRE(ctx.run() == Result::SUCCESS);
            const auto& output = ctx.output("signal");
            const F32 ref = 1.0f / (2.0f * static_cast<F32>(JST_PI));
            for (U64 channel = 0; channel < 2; ++channel) {
                const F32 increment = channel == 0 ? 0.2f : -0.35f;
                REQUIRE_THAT(output.at<F32>(0, channel, 0),
                             Catch::Matchers::WithinAbs(0.0f, 1.0e-5f));
                REQUIRE_THAT(output.at<F32>(1, channel, 0),
                             Catch::Matchers::WithinAbs(increment * ref, 1.0e-5f));
                REQUIRE_THAT(output.at<F32>(1, channel, 1),
                             Catch::Matchers::WithinAbs(increment * ref, 1.0e-5f));
            }
            REQUIRE(std::any_cast<Index>(output.attribute("batchAxis")) == Index{0});
            REQUIRE(std::any_cast<Index>(output.attribute("channelAxis")) == Index{1});
            REQUIRE(std::any_cast<Index>(output.attribute("sampleAxis")) == Index{2});
        }
    }
}

TEST_CASE("FM - Rejects unsupported or malformed signal roles",
          "[modules][fm][validation][axes]") {
    const auto implementations = Registry::ListAvailableModules("fm");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            Tensor missing;
            REQUIRE(missing.create(impl.device, DataType::CF32, {2, 4}) ==
                    Result::SUCCESS);
            RequireFmSignalValidationError(impl, missing);

            Tensor wrongType;
            REQUIRE(wrongType.create(impl.device, DataType::CF32, {2, 4}) ==
                    Result::SUCCESS);
            REQUIRE(wrongType.setAttribute("sampleAxis", I64{1}) == Result::SUCCESS);
            RequireFmSignalValidationError(impl, wrongType);

            Tensor duplicate;
            REQUIRE(duplicate.create(impl.device, DataType::CF32, {2, 4}) ==
                    Result::SUCCESS);
            REQUIRE(duplicate.setAttribute("sampleAxis", Index{1}) == Result::SUCCESS);
            REQUIRE(duplicate.setAttribute("batchAxis", Index{1}) == Result::SUCCESS);
            RequireFmSignalValidationError(impl, duplicate);

            Tensor outOfRange;
            REQUIRE(outOfRange.create(impl.device, DataType::CF32, {2, 4}) ==
                    Result::SUCCESS);
            REQUIRE(outOfRange.setAttribute("sampleAxis", Index{2}) ==
                    Result::SUCCESS);
            RequireFmSignalValidationError(impl, outOfRange);

            Tensor channelized;
            REQUIRE(channelized.create(impl.device, DataType::CF32, {2, 4}) ==
                    Result::SUCCESS);
            REQUIRE(channelized.setAttribute("sampleAxis", Index{1}) ==
                    Result::SUCCESS);
            REQUIRE(channelized.setAttribute("channelAxis", Index{0}) ==
                    Result::SUCCESS);
            Modules::FM wideConfig;
            wideConfig.mode = "wide";
            RequireFmSignalValidationError(impl, channelized, wideConfig);
        }
    }
}
