#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "jetstream/testing.hh"
#include "jetstream/registry.hh"
#include "jetstream/domains/dsp/rrc_filter/module.hh"

#include <algorithm>
#include <any>
#include <cmath>
#include <limits>
#include <type_traits>
#include <vector>

using namespace Jetstream;

namespace {

void RequireRrcFilterValidationError(const Registry::ModuleRegistration& impl,
                                     const Modules::RrcFilter& config,
                                     const DataType dtype) {
    Tensor input;
    REQUIRE(input.create(impl.device, dtype, {64}) == Result::SUCCESS);
    REQUIRE(input.setAttribute("sampleAxis", Index{0}) == Result::SUCCESS);

    TensorMap inputs;
    inputs["buffer"].requested("test", "buffer");
    inputs["buffer"].tensor = input;

    std::shared_ptr<Module> module;
    REQUIRE(Registry::BuildModule("rrc_filter", impl.device, impl.runtime,
                                  impl.provider, module) == Result::SUCCESS);
    REQUIRE(module->create("test", config, inputs) == Result::ERROR);
    REQUIRE(module->state() == Module::State::ERRORED);
    REQUIRE(module->outputs().empty());
}

void RequireRrcFilterSignalValidationError(
    const Registry::ModuleRegistration& impl,
    const Tensor& input) {
    TensorMap inputs;
    inputs["buffer"].requested("test", "buffer");
    inputs["buffer"].tensor = input;

    std::shared_ptr<Module> module;
    REQUIRE(Registry::BuildModule("rrc_filter", impl.device, impl.runtime,
                                  impl.provider, module) == Result::SUCCESS);
    REQUIRE(module->create("test", Modules::RrcFilter{}, inputs) == Result::ERROR);
    REQUIRE(module->state() == Module::State::ERRORED);
    REQUIRE(module->outputs().empty());
}

std::vector<F32> RrcImpulseResponse(const Registry::ModuleRegistration& impl,
                                    const Modules::RrcFilter& config) {
    TestContext ctx("rrc_filter", impl.device, impl.runtime, impl.provider);
    ctx.setConfig(config);
    Tensor impulse;
    REQUIRE(impulse.create(DeviceType::CPU, DataType::F32, {config.taps}) ==
            Result::SUCCESS);
    REQUIRE(impulse.setAttribute("sampleAxis", Index{0}) == Result::SUCCESS);
    std::fill_n(impulse.data<F32>(), config.taps, 0.0f);
    impulse.at<F32>(0) = 1.0f;
    ctx.setInput("buffer", impulse);
    REQUIRE(ctx.run() == Result::SUCCESS);
    const auto* coefficients = ctx.output("buffer").data<F32>();
    return {coefficients, coefficients + config.taps};
}

template<typename T>
void RequireRrcStreamingConvolution(const Registry::ModuleRegistration& impl,
                                    const U64 taps,
                                    const U64 sampleCount,
                                    const std::string& layout) {
    Modules::RrcFilter config;
    config.taps = taps;
    config.symbolRate = 1.0e6f;
    config.sampleRate = 4.0e6f;
    config.rollOff = 0.35f;
    auto coefficients = RrcImpulseResponse(impl, config);

    const bool sampleFirst = layout == "sample-first";
    const bool offsetView = layout == "offset";
    const Index sampleAxis = sampleFirst ? 0 : 2;
    const Index batchAxis = sampleFirst ? 2 : 0;
    Tensor input;
    REQUIRE(input.create(DeviceType::CPU, TypeToDataType<T>(),
                         sampleFirst ? Shape{sampleCount, 2, 2}
                                     : (offsetView ? Shape{2, 2, 2, sampleCount}
                                                   : Shape{2, 2, sampleCount})) ==
            Result::SUCCESS);
    if (offsetView) {
        REQUIRE(input.slice({Token(1), Token(), Token(), Token()}) ==
                Result::SUCCESS);
    }
    REQUIRE(input.setAttribute("sampleAxis", sampleAxis) == Result::SUCCESS);
    REQUIRE(input.setAttribute("channelAxis", Index{1}) == Result::SUCCESS);
    REQUIRE(input.setAttribute("batchAxis", batchAxis) == Result::SUCCESS);

    TestContext ctx("rrc_filter", impl.device, impl.runtime, impl.provider);
    ctx.setConfig(config);
    ctx.setInput("buffer", input);
    REQUIRE(ctx.start() == Result::SUCCESS);

    std::vector<T> streams[2];
    for (U64 frame = 0; frame < 3; ++frame) {
        if (frame == 2) {
            config.rollOff = 1.0f;
            config.sampleRate = 5.0e6f;
            REQUIRE(ctx.reconfigure(config) == Result::SUCCESS);
            coefficients = RrcImpulseResponse(impl, config);
        }
        for (U64 batch = 0; batch < 2; ++batch) {
            for (U64 channel = 0; channel < 2; ++channel) {
                for (U64 sample = 0; sample < sampleCount; ++sample) {
                    const F32 time = static_cast<F32>((frame * 2 + batch) *
                                                     sampleCount + sample);
                    const F32 real = std::sin(0.31f * time + channel) +
                                     0.3f * std::cos(0.73f * time);
                    T value;
                    if constexpr (std::is_same_v<T, CF32>) {
                        value = CF32(real, std::cos(0.17f * time + channel));
                    } else {
                        value = real;
                    }
                    const U64 offset = batch * input.stride(batchAxis) +
                                       channel * input.stride(1) +
                                       sample * input.stride(sampleAxis);
                    input.data<T>()[offset] = value;
                    streams[channel].push_back(value);
                }
            }
        }

        REQUIRE(ctx.compute() == Result::SUCCESS);
        const auto& output = ctx.output("buffer");
        for (U64 batch = 0; batch < 2; ++batch) {
            for (U64 channel = 0; channel < 2; ++channel) {
                for (U64 sample = 0; sample < sampleCount; ++sample) {
                    const U64 time = (frame * 2 + batch) * sampleCount + sample;
                    using Accumulator = std::conditional_t<std::is_same_v<T, CF32>,
                                                           CF64, F64>;
                    Accumulator expected{};
                    for (U64 k = 0; k < std::min(taps, time + 1); ++k) {
                        expected += static_cast<Accumulator>(streams[channel][time - k]) *
                                    static_cast<F64>(coefficients[k]);
                    }
                    const U64 offset = batch * output.stride(batchAxis) +
                                       channel * output.stride(1) +
                                       sample * output.stride(sampleAxis);
                    const auto actual = static_cast<Accumulator>(output.data<T>()[offset]);
                    CAPTURE(frame, batch, channel, sample);
                    REQUIRE(std::abs(actual - expected) < 2.0e-5);
                }
            }
        }
    }
    REQUIRE(ctx.stop() == Result::SUCCESS);
}

}  // namespace

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
            REQUIRE(input.setAttribute("sampleAxis", Index{0}) == Result::SUCCESS);

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
            REQUIRE(std::any_cast<Index>(out.attribute("sampleAxis")) == Index{0});
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
            REQUIRE(input.setAttribute("sampleAxis", Index{0}) == Result::SUCCESS);

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
            REQUIRE(input.setAttribute("sampleAxis", Index{0}) == Result::SUCCESS);

            ctx.setInput("buffer", input);

            REQUIRE(ctx.run() != Result::SUCCESS);
        }
    }
}

TEST_CASE("RRC Filter - Matches streaming convolution across layouts and reconfiguration",
          "[modules][rrc_filter][streaming]") {
    const auto implementations = Registry::ListAvailableModules("rrc_filter");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        for (const U64 taps : {3, 11, 101}) {
            for (const U64 samples : {U64{1}, taps - 1, taps, taps + 7,
                                      taps + 31, taps + 32, U64{257}}) {
                for (const std::string layout : {"contiguous", "sample-first", "offset"}) {
                    DYNAMIC_SECTION("Device: " << impl.device << " Runtime: "
                                    << impl.runtime << " taps: " << taps
                                    << " samples: " << samples << " layout: " << layout) {
                        RequireRrcStreamingConvolution<F32>(impl, taps, samples, layout);
                        RequireRrcStreamingConvolution<CF32>(impl, taps, samples, layout);
                    }
                }
            }
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
            REQUIRE(input.setAttribute("sampleAxis", Index{0}) == Result::SUCCESS);

            ctx.setInput("buffer", input);

            REQUIRE(ctx.run() != Result::SUCCESS);
        }
    }
}

TEST_CASE("RRC Filter - Keeps channels independent for either sample-axis placement",
          "[modules][rrc_filter][axes]") {
    const auto implementations = Registry::ListAvailableModules("rrc_filter");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        for (const bool sampleFirst : {false, true}) {
            DYNAMIC_SECTION("Device: " << impl.device << " Runtime: "
                            << impl.runtime << " sampleFirst: " << sampleFirst) {
                TestContext ctx("rrc_filter", impl.device, impl.runtime,
                                impl.provider);
                Modules::RrcFilter config;
                config.taps = 5;
                ctx.setConfig(config);

                Tensor input;
                REQUIRE(input.create(DeviceType::CPU, DataType::F32,
                                     sampleFirst ? Shape{8, 2} : Shape{2, 8}) ==
                        Result::SUCCESS);
                const Index sampleAxis = sampleFirst ? 0 : 1;
                const Index channelAxis = sampleFirst ? 1 : 0;
                REQUIRE(input.setAttribute("sampleAxis", sampleAxis) ==
                        Result::SUCCESS);
                REQUIRE(input.setAttribute("channelAxis", channelAxis) ==
                        Result::SUCCESS);
                std::fill(input.data<F32>(), input.data<F32>() + input.size(), 0.0f);
                if (sampleFirst) {
                    input.at<F32>(7, 0) = 1.0f;
                } else {
                    input.at<F32>(0, 7) = 1.0f;
                }
                ctx.setInput("buffer", input);

                REQUIRE(ctx.run() == Result::SUCCESS);
                const auto& output = ctx.output("buffer");
                for (U64 sample = 0; sample < 8; ++sample) {
                    const F32 value = sampleFirst
                        ? output.at<F32>(sample, 1)
                        : output.at<F32>(1, sample);
                    REQUIRE_THAT(value,
                                 Catch::Matchers::WithinAbs(0.0f, 1.0e-6f));
                }
                REQUIRE(std::any_cast<Index>(output.attribute("sampleAxis")) ==
                        sampleAxis);
                REQUIRE(std::any_cast<Index>(output.attribute("channelAxis")) ==
                        channelAxis);
            }
        }
    }
}

TEST_CASE("RRC Filter - Sequences batches independently per channel",
          "[modules][rrc_filter][axes][batch]") {
    const auto implementations = Registry::ListAvailableModules("rrc_filter");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            TestContext ctx("rrc_filter", impl.device, impl.runtime, impl.provider);
            Modules::RrcFilter config;
            config.taps = 5;
            ctx.setConfig(config);

            Tensor input;
            REQUIRE(input.create(DeviceType::CPU, DataType::F32, {2, 2, 4}) ==
                    Result::SUCCESS);
            REQUIRE(input.setAttribute("batchAxis", Index{0}) == Result::SUCCESS);
            REQUIRE(input.setAttribute("channelAxis", Index{1}) == Result::SUCCESS);
            REQUIRE(input.setAttribute("sampleAxis", Index{2}) == Result::SUCCESS);
            std::fill(input.data<F32>(), input.data<F32>() + input.size(), 0.0f);
            input.at<F32>(0, 0, 3) = 1.0f;
            ctx.setInput("buffer", input);

            REQUIRE(ctx.run() == Result::SUCCESS);
            const auto& output = ctx.output("buffer");
            REQUIRE(std::abs(output.at<F32>(1, 0, 0)) > 1.0e-4f);
            for (U64 batch = 0; batch < 2; ++batch) {
                for (U64 sample = 0; sample < 4; ++sample) {
                    REQUIRE_THAT(output.at<F32>(batch, 1, sample),
                                 Catch::Matchers::WithinAbs(0.0f, 1.0e-6f));
                }
            }
            REQUIRE(std::any_cast<Index>(output.attribute("batchAxis")) == Index{0});
            REQUIRE(std::any_cast<Index>(output.attribute("channelAxis")) == Index{1});
            REQUIRE(std::any_cast<Index>(output.attribute("sampleAxis")) == Index{2});
        }
    }
}

TEST_CASE("RRC Filter - Rejects missing or malformed signal roles",
          "[modules][rrc_filter][validation][axes]") {
    const auto implementations = Registry::ListAvailableModules("rrc_filter");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            Tensor missing;
            REQUIRE(missing.create(impl.device, DataType::F32, {2, 4}) ==
                    Result::SUCCESS);
            RequireRrcFilterSignalValidationError(impl, missing);

            Tensor wrongType;
            REQUIRE(wrongType.create(impl.device, DataType::F32, {2, 4}) ==
                    Result::SUCCESS);
            REQUIRE(wrongType.setAttribute("sampleAxis", I64{1}) == Result::SUCCESS);
            RequireRrcFilterSignalValidationError(impl, wrongType);

            Tensor duplicate;
            REQUIRE(duplicate.create(impl.device, DataType::F32, {2, 4}) ==
                    Result::SUCCESS);
            REQUIRE(duplicate.setAttribute("sampleAxis", Index{1}) == Result::SUCCESS);
            REQUIRE(duplicate.setAttribute("channelAxis", Index{1}) ==
                    Result::SUCCESS);
            RequireRrcFilterSignalValidationError(impl, duplicate);

            Tensor outOfRange;
            REQUIRE(outOfRange.create(impl.device, DataType::F32, {2, 4}) ==
                    Result::SUCCESS);
            REQUIRE(outOfRange.setAttribute("sampleAxis", Index{2}) ==
                    Result::SUCCESS);
            RequireRrcFilterSignalValidationError(impl, outOfRange);
        }
    }
}

TEST_CASE("RRC Filter - Rejects invalid candidates during validation",
          "[modules][rrc_filter][validation]") {
    auto implementations = Registry::ListAvailableModules("rrc_filter");
    REQUIRE(!implementations.empty());

    for (const auto& impl : implementations) {
        DYNAMIC_SECTION("Device: " << impl.device << " Runtime: " << impl.runtime) {
            SECTION("numeric configuration must be finite") {
                Modules::RrcFilter config;
                config.symbolRate = std::numeric_limits<F32>::quiet_NaN();
                RequireRrcFilterValidationError(impl, config, DataType::F32);
            }

            SECTION("symbol rate must be positive") {
                Modules::RrcFilter config;
                config.symbolRate = 0.0f;
                RequireRrcFilterValidationError(impl, config, DataType::F32);
            }

            SECTION("sample rate must be positive") {
                Modules::RrcFilter config;
                config.sampleRate = 0.0f;
                RequireRrcFilterValidationError(impl, config, DataType::F32);
            }

            SECTION("roll-off must be finite") {
                Modules::RrcFilter config;
                config.rollOff = std::numeric_limits<F32>::quiet_NaN();
                RequireRrcFilterValidationError(impl, config, DataType::F32);
            }

            SECTION("roll-off cannot be negative") {
                Modules::RrcFilter config;
                config.rollOff = -0.01f;
                RequireRrcFilterValidationError(impl, config, DataType::F32);
            }

            SECTION("roll-off cannot exceed one") {
                Modules::RrcFilter config;
                config.rollOff = 1.01f;
                RequireRrcFilterValidationError(impl, config, DataType::F32);
            }

            SECTION("tap count must be at least three") {
                Modules::RrcFilter config;
                config.taps = 1;
                RequireRrcFilterValidationError(impl, config, DataType::F32);
            }

            SECTION("input dtype must be supported") {
                Modules::RrcFilter config;
                RequireRrcFilterValidationError(impl, config, DataType::U8);
            }

            SECTION("tap buffer size must be representable") {
                Modules::RrcFilter config;
                config.taps = std::numeric_limits<U64>::max();
                RequireRrcFilterValidationError(impl, config, DataType::F32);
            }
        }
    }
}
