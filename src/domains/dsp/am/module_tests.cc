#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "jetstream/testing.hh"
#include "jetstream/registry.hh"
#include "jetstream/domains/dsp/am/module.hh"

#include <algorithm>
#include <any>
#include <cmath>
#include <limits>

using namespace Jetstream;

namespace {

void RequireAmValidationError(const Registry::ModuleRegistration& impl,
                              const Modules::AM& config,
                              const DataType dtype) {
    Tensor input;
    REQUIRE(input.create(impl.device, dtype, {16}) == Result::SUCCESS);
    REQUIRE(input.setAttribute("sampleAxis", Index{0}) == Result::SUCCESS);

    TensorMap inputs;
    inputs["signal"].requested("test", "signal");
    inputs["signal"].tensor = input;

    std::shared_ptr<Module> module;
    REQUIRE(Registry::BuildModule("am", impl.device, impl.runtime,
                                  impl.provider, module) == Result::SUCCESS);
    REQUIRE(module->create("test", config, inputs) == Result::ERROR);
    REQUIRE(module->state() == Module::State::ERRORED);
    REQUIRE(module->outputs().empty());
}

void RequireAmSignalValidationError(const Registry::ModuleRegistration& impl,
                                    const Tensor& input) {
    TensorMap inputs;
    inputs["signal"].requested("test", "signal");
    inputs["signal"].tensor = input;

    std::shared_ptr<Module> module;
    REQUIRE(Registry::BuildModule("am", impl.device, impl.runtime,
                                  impl.provider, module) == Result::SUCCESS);
    REQUIRE(module->create("test", Modules::AM{}, inputs) == Result::ERROR);
    REQUIRE(module->state() == Module::State::ERRORED);
    REQUIRE(module->outputs().empty());
}

}  // namespace

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
            REQUIRE(input.setAttribute("sampleAxis", Index{0}) == Result::SUCCESS);

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

TEST_CASE("AM - Validation Rejects Invalid Configuration Before Create",
          "[modules][am][validation]") {
    auto implementations = Registry::ListAvailableModules("am");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            SECTION("sample rate must be finite and positive") {
                Modules::AM config;
                config.sampleRate = 0.0f;
                RequireAmValidationError(impl, config, DataType::CF32);

                config.sampleRate = std::numeric_limits<F32>::quiet_NaN();
                RequireAmValidationError(impl, config, DataType::CF32);
            }

            SECTION("DC alpha must be finite and in range") {
                Modules::AM config;
                for (const F32 dcAlpha : {
                         -0.1f,
                         1.0f,
                         std::numeric_limits<F32>::quiet_NaN(),
                     }) {
                    config.dcAlpha = dcAlpha;
                    RequireAmValidationError(impl, config, DataType::CF32);
                }
            }

            SECTION("native CPU input must be CF32") {
                Modules::AM config;
                RequireAmValidationError(impl, config, DataType::F32);
            }
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
            REQUIRE(input.setAttribute("sampleAxis", Index{0}) == Result::SUCCESS);

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
            REQUIRE(input.setAttribute("sampleAxis", Index{0}) == Result::SUCCESS);

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

TEST_CASE("AM - Publishes baseband frequency metadata",
          "[modules][am][metadata]") {
    const auto implementations = Registry::ListAvailableModules("am");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            TestContext ctx("am", impl.device, impl.runtime, impl.provider);
            ctx.setConfig(Modules::AM{});

            Tensor input;
            REQUIRE(input.create(DeviceType::CPU, DataType::CF32, {16}) ==
                    Result::SUCCESS);
            REQUIRE(input.setAttribute("sampleAxis", Index{0}) == Result::SUCCESS);
            std::fill(input.data<CF32>(), input.data<CF32>() + input.size(),
                      CF32{0.0f, 0.0f});
            REQUIRE(input.setAttribute("frequency", F32{100.0e6f}) ==
                    Result::SUCCESS);
            REQUIRE(input.setAttribute("sampleRate", F32{240.0e3f}) ==
                    Result::SUCCESS);
            ctx.setInput("signal", input);

            REQUIRE(ctx.run() == Result::SUCCESS);
            const auto& output = ctx.output("signal");
            REQUIRE(std::any_cast<F32>(output.attribute("frequency")) == 0.0f);
            REQUIRE(std::any_cast<F32>(output.attribute("sampleRate")) ==
                    240.0e3f);
            REQUIRE(std::any_cast<Index>(output.attribute("sampleAxis")) ==
                    Index{0});
        }
    }
}

TEST_CASE("AM - Keeps channels independent for either sample-axis placement",
          "[modules][am][axes]") {
    const auto implementations = Registry::ListAvailableModules("am");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        for (const bool sampleFirst : {false, true}) {
            DYNAMIC_SECTION("Device: " << impl.device << " Runtime: "
                            << impl.runtime << " sampleFirst: " << sampleFirst) {
                TestContext ctx("am", impl.device, impl.runtime, impl.provider);
                Modules::AM config;
                config.dcAlpha = 0.0f;
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
                    const F32 envelope = channel == 0 ? 1.0f : 10.0f;
                    for (U64 sample = 0; sample < 4; ++sample) {
                        if (sampleFirst) {
                            input.at<CF32>(sample, channel) = CF32{envelope, 0.0f};
                        } else {
                            input.at<CF32>(channel, sample) = CF32{envelope, 0.0f};
                        }
                    }
                }
                ctx.setInput("signal", input);

                REQUIRE(ctx.run() == Result::SUCCESS);
                const auto& output = ctx.output("signal");
                for (U64 channel = 0; channel < 2; ++channel) {
                    const F32 envelope = channel == 0 ? 1.0f : 10.0f;
                    for (U64 sample = 0; sample < 4; ++sample) {
                        const F32 value = sampleFirst
                            ? output.at<F32>(sample, channel)
                            : output.at<F32>(channel, sample);
                        REQUIRE_THAT(value, Catch::Matchers::WithinAbs(
                            sample == 0 ? envelope : 0.0f, 1.0e-6f));
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

TEST_CASE("AM - Sequences batches independently per channel",
          "[modules][am][axes][batch]") {
    const auto implementations = Registry::ListAvailableModules("am");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            TestContext ctx("am", impl.device, impl.runtime, impl.provider);
            Modules::AM config;
            config.dcAlpha = 0.0f;
            ctx.setConfig(config);

            Tensor input;
            REQUIRE(input.create(DeviceType::CPU, DataType::CF32, {2, 2, 2}) ==
                    Result::SUCCESS);
            REQUIRE(input.setAttribute("batchAxis", Index{0}) == Result::SUCCESS);
            REQUIRE(input.setAttribute("channelAxis", Index{1}) == Result::SUCCESS);
            REQUIRE(input.setAttribute("sampleAxis", Index{2}) == Result::SUCCESS);

            input.at<CF32>(0, 0, 0) = CF32{1.0f, 0.0f};
            input.at<CF32>(0, 0, 1) = CF32{1.0f, 0.0f};
            input.at<CF32>(1, 0, 0) = CF32{1.0f, 0.0f};
            input.at<CF32>(1, 0, 1) = CF32{3.0f, 0.0f};
            input.at<CF32>(0, 1, 0) = CF32{10.0f, 0.0f};
            input.at<CF32>(0, 1, 1) = CF32{10.0f, 0.0f};
            input.at<CF32>(1, 1, 0) = CF32{10.0f, 0.0f};
            input.at<CF32>(1, 1, 1) = CF32{13.0f, 0.0f};
            ctx.setInput("signal", input);

            REQUIRE(ctx.run() == Result::SUCCESS);
            const auto& output = ctx.output("signal");
            REQUIRE_THAT(output.at<F32>(0, 0, 0),
                         Catch::Matchers::WithinAbs(1.0f, 1.0e-6f));
            REQUIRE_THAT(output.at<F32>(0, 1, 0),
                         Catch::Matchers::WithinAbs(10.0f, 1.0e-6f));
            REQUIRE_THAT(output.at<F32>(1, 0, 0),
                         Catch::Matchers::WithinAbs(0.0f, 1.0e-6f));
            REQUIRE_THAT(output.at<F32>(1, 0, 1),
                         Catch::Matchers::WithinAbs(2.0f, 1.0e-6f));
            REQUIRE_THAT(output.at<F32>(1, 1, 0),
                         Catch::Matchers::WithinAbs(0.0f, 1.0e-6f));
            REQUIRE_THAT(output.at<F32>(1, 1, 1),
                         Catch::Matchers::WithinAbs(3.0f, 1.0e-6f));
            REQUIRE(std::any_cast<Index>(output.attribute("batchAxis")) == Index{0});
            REQUIRE(std::any_cast<Index>(output.attribute("channelAxis")) == Index{1});
            REQUIRE(std::any_cast<Index>(output.attribute("sampleAxis")) == Index{2});
        }
    }
}

TEST_CASE("AM - Rejects missing or malformed signal roles",
          "[modules][am][validation][axes]") {
    const auto implementations = Registry::ListAvailableModules("am");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            Tensor missing;
            REQUIRE(missing.create(impl.device, DataType::CF32, {2, 4}) ==
                    Result::SUCCESS);
            RequireAmSignalValidationError(impl, missing);

            Tensor wrongType;
            REQUIRE(wrongType.create(impl.device, DataType::CF32, {2, 4}) ==
                    Result::SUCCESS);
            REQUIRE(wrongType.setAttribute("sampleAxis", I64{1}) == Result::SUCCESS);
            RequireAmSignalValidationError(impl, wrongType);

            Tensor duplicate;
            REQUIRE(duplicate.create(impl.device, DataType::CF32, {2, 4}) ==
                    Result::SUCCESS);
            REQUIRE(duplicate.setAttribute("sampleAxis", Index{1}) == Result::SUCCESS);
            REQUIRE(duplicate.setAttribute("channelAxis", Index{1}) ==
                    Result::SUCCESS);
            RequireAmSignalValidationError(impl, duplicate);

            Tensor outOfRange;
            REQUIRE(outOfRange.create(impl.device, DataType::CF32, {2, 4}) ==
                    Result::SUCCESS);
            REQUIRE(outOfRange.setAttribute("sampleAxis", Index{2}) ==
                    Result::SUCCESS);
            RequireAmSignalValidationError(impl, outOfRange);
        }
    }
}
