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
                                    const Tensor& input) {
    TensorMap inputs;
    inputs["signal"].requested("test", "signal");
    inputs["signal"].tensor = input;

    std::shared_ptr<Module> module;
    REQUIRE(Registry::BuildModule("fm", impl.device, impl.runtime,
                                  impl.provider, module) == Result::SUCCESS);
    REQUIRE(module->create("test", Modules::FM{}, inputs) == Result::ERROR);
    REQUIRE(module->state() == Module::State::ERRORED);
    REQUIRE(module->outputs().empty());
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
            SECTION("sample rate must be finite and positive") {
                Modules::FM config;
                config.sampleRate = 0.0f;
                RequireFmValidationError(impl, config, DataType::CF32);

                config.sampleRate = std::numeric_limits<F32>::quiet_NaN();
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

TEST_CASE("FM - Rejects missing or malformed signal roles",
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
        }
    }
}
