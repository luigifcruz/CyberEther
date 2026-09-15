#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <algorithm>
#include <any>
#include <array>
#include <cmath>
#include <limits>
#include <string>
#include <type_traits>
#include <unordered_set>
#include <utility>
#include <vector>

#include "jetstream/domains/dsp/rational_resampler/module.hh"
#include "jetstream/registry.hh"
#include "jetstream/runtime.hh"
#include "jetstream/testing.hh"

using namespace Jetstream;

namespace {

TensorMap ResamplerInput(const Tensor& input) {
    TensorMap ports;
    ports["buffer"].requested("source", "buffer");
    ports["buffer"].tensor = input;
    return ports;
}

std::shared_ptr<Module> CreateResampler(const Modules::RationalResampler& config,
                                       const Tensor& input) {
    std::shared_ptr<Module> module;
    REQUIRE(Registry::BuildModule("rational_resampler", DeviceType::CPU,
                                  RuntimeType::NATIVE, "generic", module) == Result::SUCCESS);
    REQUIRE(module->create("test", config, ResamplerInput(input)) == Result::SUCCESS);
    return module;
}

void RequireValidationError(const Modules::RationalResampler& config,
                            const Tensor& input) {
    std::shared_ptr<Module> module;
    REQUIRE(Registry::BuildModule("rational_resampler", DeviceType::CPU,
                                  RuntimeType::NATIVE, "generic", module) == Result::SUCCESS);
    REQUIRE(module->create("test", config, ResamplerInput(input)) == Result::ERROR);
    REQUIRE(module->state() == Module::State::ERRORED);
    REQUIRE(module->outputs().empty());
}

template<typename T>
T StreamSample(const U64 time, const U64 lane) {
    if (lane == 1) {
        return T{};
    }
    const F32 real = std::sin(0.17 * time + lane) + 0.3 * std::cos(0.41 * time);
    if constexpr (std::is_same_v<T, CF32>) {
        return {real, static_cast<F32>(std::cos(0.13 * time + lane))};
    } else {
        return real;
    }
}

template<typename T>
void RequireStreaming(const Modules::RationalResampler& config, const U64 samples,
                      const std::string& layout) {
    constexpr U64 frames = 96;
    constexpr U64 batches = 2;
    constexpr U64 lanes = 3;
    const U64 totalSamples = frames * batches * samples;
    Tensor referenceInput;
    REQUIRE(referenceInput.create(DeviceType::CPU, TypeToDataType<T>(),
                                   {lanes, totalSamples}) == Result::SUCCESS);
    REQUIRE(referenceInput.setAttribute("sampleAxis", Index{1}) == Result::SUCCESS);
    REQUIRE(referenceInput.setAttribute("channelAxis", Index{0}) == Result::SUCCESS);
    for (U64 lane = 0; lane < lanes; ++lane) {
        for (U64 i = 0; i < totalSamples; ++i) {
            referenceInput.at<T>(lane, i) = StreamSample<T>(i, lane);
        }
    }
    TestContext reference("rational_resampler", DeviceType::CPU,
                          RuntimeType::NATIVE, "generic");
    reference.setConfig(config);
    reference.setInput("buffer", referenceInput);
    REQUIRE(reference.run() == Result::SUCCESS);
    const auto& expected = reference.output("buffer");

    const bool sampleFirst = layout == "sample-first";
    const bool strided = layout == "strided";
    const Index sampleAxis = sampleFirst ? 0 : 2;
    const Index batchAxis = sampleFirst ? 2 : 0;
    Tensor input;
    REQUIRE(input.create(DeviceType::CPU, TypeToDataType<T>(),
                         sampleFirst ? Shape{samples, lanes, batches} :
                         Shape{batches, lanes, strided ? 2 * samples + 1 : samples}) ==
            Result::SUCCESS);
    if (strided) {
        REQUIRE(input.slice({Token(), Token(), Token(U64{1}, 2 * samples + 1, U64{2})}) ==
                Result::SUCCESS);
        REQUIRE(input.offset() > 0);
        REQUIRE_FALSE(input.contiguous());
    }
    REQUIRE(input.setAttribute("sampleAxis", sampleAxis) == Result::SUCCESS);
    REQUIRE(input.setAttribute("batchAxis", batchAxis) == Result::SUCCESS);
    REQUIRE(input.setAttribute("channelAxis", Index{1}) == Result::SUCCESS);
    auto module = CreateResampler(config, input);
    Runtime runtime("test", DeviceType::CPU, RuntimeType::NATIVE);
    REQUIRE(runtime.create({{"test", module}}) == Result::SUCCESS);
    const auto& output = module->outputs().at("buffer").tensor;
    const U64 outputSamples = output.shape(sampleAxis);
    const U64 quantum = batches * outputSamples;
    U64 emitted = 0;
    for (U64 frame = 0; frame < frames; ++frame) {
        for (U64 batch = 0; batch < batches; ++batch) {
            for (U64 lane = 0; lane < lanes; ++lane) {
                for (U64 sample = 0; sample < samples; ++sample) {
                    const U64 time = (frame * batches + batch) * samples + sample;
                    const U64 offset = batch * input.stride(batchAxis) +
                                       lane * input.stride(1) +
                                       sample * input.stride(sampleAxis);
                    input.data<T>()[offset] = StreamSample<T>(time, lane);
                }
            }
        }
        std::unordered_set<std::string> skipped;
        std::unordered_set<std::string> failed;
        REQUIRE(runtime.compute({}, skipped, failed) == Result::SUCCESS);
        REQUIRE(failed.empty());
        const U64 ticks = (frame + 1) * batches * samples * config.interpolation;
        const U64 available = (ticks + config.decimation - 1) / config.decimation;
        CAPTURE(frame, samples, layout);
        REQUIRE(skipped.contains("test") == (available - emitted < quantum));
        if (!skipped.empty()) {
            continue;
        }
        for (U64 batch = 0; batch < batches; ++batch) {
            for (U64 lane = 0; lane < lanes; ++lane) {
                for (U64 sample = 0; sample < outputSamples; ++sample) {
                    const U64 offset = batch * output.stride(batchAxis) +
                                       lane * output.stride(1) +
                                       sample * output.stride(sampleAxis);
                    const U64 index = emitted + batch * outputSamples + sample;
                    const T value = expected.at<T>(lane, index);
                    REQUIRE(std::abs(output.data<T>()[offset] - value) < 2.0e-6f);
                }
            }
        }
        emitted += quantum;
    }
    REQUIRE(emitted > 0);
    REQUIRE(expected.shape(1) - emitted < quantum);
    REQUIRE(std::any_cast<Index>(output.attribute("sampleAxis")) == sampleAxis);
    REQUIRE(std::any_cast<Index>(output.attribute("batchAxis")) == batchAxis);
    REQUIRE(std::any_cast<Index>(output.attribute("channelAxis")) == Index{1});
    REQUIRE(runtime.destroy() == Result::SUCCESS);
    REQUIRE(module->destroy() == Result::SUCCESS);
}

}  // namespace

TEST_CASE("Rational Resampler - CPU registration", "[modules][rational_resampler]") {
    const auto implementations = Registry::ListAvailableModules("rational_resampler");
    REQUIRE(implementations.size() == 1);
    REQUIRE(implementations.front().device == DeviceType::CPU);
    REQUIRE(implementations.front().runtime == RuntimeType::NATIVE);
}

TEST_CASE("Rational Resampler - Default filter preserves gain and suppresses aliases and images",
          "[modules][rational_resampler][frequency]") {
    const std::array<std::pair<U64, U64>, 4> ratios{{{1, 4}, {3, 2}, {24, 125}, {1, 1}}};
    for (const auto& [l, m] : ratios) {
        for (const bool stopband : {false, true}) {
            CAPTURE(l, m, stopband);
            const F64 ratio = static_cast<F64>(l) / m;
            const F64 frequency = stopband ? 0.49 : 0.05 * std::min(1.0, ratio);
            Tensor input;
            REQUIRE(input.create(DeviceType::CPU, DataType::CF32, {8000}) == Result::SUCCESS);
            REQUIRE(input.setAttribute("sampleAxis", Index{0}) == Result::SUCCESS);
            for (U64 i = 0; i < input.size(); ++i) {
                const F64 phase = 2.0 * JST_PI * frequency * i;
                input.at<CF32>(i) = static_cast<CF32>(std::polar(1.0, phase));
            }
            Modules::RationalResampler config;
            config.interpolation = l;
            config.decimation = m;
            TestContext ctx("rational_resampler", DeviceType::CPU,
                            RuntimeType::NATIVE, "generic");
            ctx.setConfig(config);
            ctx.setInput("buffer", input);
            REQUIRE(ctx.run() == Result::SUCCESS);
            const auto& output = ctx.output("buffer");
            const F64 delay = 32.0 * std::max(l, m) / l;
            F64 errorPower = 0;
            for (U64 i = 256; i < output.size(); ++i) {
                const CF64 expected = stopband ? CF64{} :
                    std::polar(1.0, 2.0 * JST_PI * frequency * (i / ratio - delay));
                errorPower += std::norm(static_cast<CF64>(output.at<CF32>(i)) - expected);
            }
            REQUIRE(std::sqrt(errorPower / (output.size() - 256)) < 0.001);
        }
    }
}

TEST_CASE("Rational Resampler - Reduces large factors and handles tiny cutoffs",
          "[modules][rational_resampler][numeric]") {
    Tensor input;
    REQUIRE(input.create(DeviceType::CPU, DataType::F32, {256}) == Result::SUCCESS);
    REQUIRE(input.setAttribute("sampleAxis", Index{0}) == Result::SUCCESS);
    std::fill_n(input.data<F32>(), input.size(), 1.0f);
    Modules::RationalResampler config;
    config.interpolation = std::numeric_limits<U64>::max();
    config.decimation = config.interpolation;
    config.cutoff = std::numeric_limits<F32>::min();
    TestContext ctx("rational_resampler", DeviceType::CPU, RuntimeType::NATIVE, "generic");
    ctx.setConfig(config);
    ctx.setInput("buffer", input);
    REQUIRE(ctx.run() == Result::SUCCESS);
    const auto& output = ctx.output("buffer");
    REQUIRE(output.shape() == input.shape());
    for (U64 i = 64; i < output.size(); ++i) {
        REQUIRE_THAT(output.at<F32>(i), Catch::Matchers::WithinAbs(1.0f, 1.0e-6f));
    }
}

TEST_CASE("Rational Resampler - Chains before sample rate is published",
          "[modules][rational_resampler][metadata][late-metadata]") {
    Tensor input;
    REQUIRE(input.create(DeviceType::CPU, DataType::F32, {96}) == Result::SUCCESS);
    REQUIRE(input.setAttribute("sampleAxis", Index{0}) == Result::SUCCESS);
    REQUIRE_FALSE(input.hasAttribute("sampleRate"));
    std::fill_n(input.data<F32>(), input.size(), 1.0f);

    Modules::RationalResampler config;
    config.interpolation = 2;
    config.decimation = 3;
    auto module = CreateResampler(config, input);
    const auto& output = module->outputs().at("buffer").tensor;
    config.interpolation = 3;
    config.decimation = 2;
    auto downstream = CreateResampler(config, output);
    const auto& downstreamOutput = downstream->outputs().at("buffer").tensor;
    Runtime upstreamRuntime("upstream", DeviceType::CPU, RuntimeType::NATIVE);
    Runtime downstreamRuntime("downstream", DeviceType::CPU, RuntimeType::NATIVE);
    REQUIRE(upstreamRuntime.create({{"test", module}}) == Result::SUCCESS);
    REQUIRE(downstreamRuntime.create({{"test", downstream}}) == Result::SUCCESS);
    const auto computeChain = [&]() {
        std::unordered_set<std::string> skipped;
        std::unordered_set<std::string> failed;
        REQUIRE(upstreamRuntime.compute({}, skipped, failed) == Result::SUCCESS);
        REQUIRE(skipped.empty());
        REQUIRE(failed.empty());
        REQUIRE(downstreamRuntime.compute({}, skipped, failed) == Result::SUCCESS);
        REQUIRE(skipped.empty());
        REQUIRE(failed.empty());
    };
    REQUIRE_FALSE(output.hasAttribute("sampleRate"));
    REQUIRE_FALSE(output.attribute("sampleRate").has_value());
    REQUIRE(output.attributeKeys() == std::vector<std::string>{"sampleAxis"});
    REQUIRE_FALSE(downstreamOutput.hasAttribute("sampleRate"));
    REQUIRE_FALSE(downstreamOutput.attribute("sampleRate").has_value());
    REQUIRE(downstreamOutput.attributeKeys() == std::vector<std::string>{"sampleAxis"});
    computeChain();
    REQUIRE_FALSE(output.hasAttribute("sampleRate"));
    REQUIRE_FALSE(downstreamOutput.hasAttribute("sampleRate"));

    REQUIRE(input.setAttribute("sampleRate", F32{48000}) == Result::SUCCESS);
    REQUIRE_FALSE(output.hasAttribute("sampleRate"));
    REQUIRE_FALSE(downstreamOutput.hasAttribute("sampleRate"));
    computeChain();
    REQUIRE(output.hasAttribute("sampleRate"));
    REQUIRE(downstreamOutput.hasAttribute("sampleRate"));
    REQUIRE(std::any_cast<F32>(output.attribute("sampleRate")) == 32000.0f);
    REQUIRE(std::any_cast<F32>(downstreamOutput.attribute("sampleRate")) == 48000.0f);
    REQUIRE(output.attributeKeys() ==
            std::vector<std::string>{"sampleAxis", "sampleRate"});
    REQUIRE(downstreamOutput.attributeKeys() == output.attributeKeys());
    REQUIRE(input.setAttribute("sampleRate", F32{96000}) == Result::SUCCESS);
    computeChain();
    REQUIRE(std::any_cast<F32>(output.attribute("sampleRate")) == 64000.0f);
    REQUIRE(std::any_cast<F32>(downstreamOutput.attribute("sampleRate")) == 96000.0f);
    REQUIRE(input.removeAttribute("sampleRate") == Result::SUCCESS);
    computeChain();
    REQUIRE_FALSE(output.hasAttribute("sampleRate"));
    REQUIRE_FALSE(output.attribute("sampleRate").has_value());
    REQUIRE_FALSE(downstreamOutput.hasAttribute("sampleRate"));
    REQUIRE_FALSE(downstreamOutput.attribute("sampleRate").has_value());
    REQUIRE(output.attributeKeys() == std::vector<std::string>{"sampleAxis"});
    REQUIRE(downstreamOutput.attributeKeys() == output.attributeKeys());
    REQUIRE(input.setAttribute("sampleRate", F32{48000}) == Result::SUCCESS);
    REQUIRE_FALSE(output.hasAttribute("sampleRate"));
    REQUIRE_FALSE(downstreamOutput.hasAttribute("sampleRate"));
    computeChain();
    REQUIRE(output.hasAttribute("sampleRate"));
    REQUIRE(downstreamOutput.hasAttribute("sampleRate"));
    REQUIRE(std::any_cast<F32>(output.attribute("sampleRate")) == 32000.0f);
    REQUIRE(std::any_cast<F32>(downstreamOutput.attribute("sampleRate")) == 48000.0f);
    REQUIRE(downstreamRuntime.destroy() == Result::SUCCESS);
    REQUIRE(upstreamRuntime.destroy() == Result::SUCCESS);
    REQUIRE(downstream->destroy() == Result::SUCCESS);
    REQUIRE(module->destroy() == Result::SUCCESS);
}

TEST_CASE("Rational Resampler - Preserves metadata and validates edits without changing state",
          "[modules][rational_resampler][metadata][reconfigure]") {
    Tensor input;
    REQUIRE(input.create(DeviceType::CPU, DataType::F32, {125}) == Result::SUCCESS);
    REQUIRE(input.setAttribute("sampleAxis", Index{0}) == Result::SUCCESS);
    REQUIRE(input.setAttribute("sampleRate", F32{250000}) == Result::SUCCESS);
    REQUIRE(input.setAttribute("frequency", F32{100000000}) == Result::SUCCESS);
    std::fill_n(input.data<F32>(), input.size(), 1.0f);
    Modules::RationalResampler config;
    config.interpolation = 24;
    config.decimation = 125;
    TestContext ctx("rational_resampler", DeviceType::CPU, RuntimeType::NATIVE, "generic");
    ctx.setConfig(config);
    ctx.setInput("buffer", input);
    REQUIRE(ctx.start() == Result::SUCCESS);
    REQUIRE(ctx.compute() == Result::SUCCESS);
    const auto& output = ctx.output("buffer");
    REQUIRE(output.shape() == Shape{24});
    REQUIRE(std::any_cast<F32>(output.attribute("sampleRate")) == 48000.0f);
    REQUIRE(std::any_cast<F32>(output.attribute("frequency")) == 100000000.0f);
    REQUIRE(input.setAttribute("sampleRate", F32{500000}) == Result::SUCCESS);
    REQUIRE(ctx.compute() == Result::SUCCESS);
    REQUIRE(std::any_cast<F32>(output.attribute("sampleRate")) == 96000.0f);
    auto changed = config;
    changed.interpolation = 3;
    changed.decimation = 2;
    REQUIRE(ctx.reconfigure(changed, true) == Result::SUCCESS);
    REQUIRE(ctx.compute() == Result::SUCCESS);
    REQUIRE(ctx.output("buffer").shape() == Shape{24});
    REQUIRE(ctx.reconfigure(changed) == Result::RECREATE);
    changed.decimation = 0;
    REQUIRE(ctx.reconfigure(changed, true) == Result::ERROR);
    REQUIRE(ctx.compute() == Result::SUCCESS);
    REQUIRE(ctx.output("buffer").shape() == Shape{24});
    REQUIRE(ctx.stop() == Result::SUCCESS);
}

TEST_CASE("Rational Resampler - Validates configuration before allocation",
          "[modules][rational_resampler][validation]") {
    Tensor input;
    REQUIRE(input.create(DeviceType::CPU, DataType::F32, {64}) == Result::SUCCESS);
    REQUIRE(input.setAttribute("sampleAxis", Index{0}) == Result::SUCCESS);
    Modules::RationalResampler config;
    SECTION("zero interpolation") { config.interpolation = 0; }
    SECTION("zero decimation") { config.decimation = 0; }
    SECTION("even taps") { config.taps = 12; }
    SECTION("insufficient taps") { config.interpolation = 5; config.taps = 9; }
    SECTION("zero cutoff") { config.cutoff = 0; }
    SECTION("negative cutoff") { config.cutoff = -0.1f; }
    SECTION("Nyquist cutoff") { config.cutoff = 1; }
    SECTION("nonfinite cutoff") { config.cutoff = std::numeric_limits<F32>::quiet_NaN(); }
    SECTION("overflowing ratio") { config.interpolation = std::numeric_limits<U64>::max(); }
    SECTION("overflowing taps") { config.taps = std::numeric_limits<U64>::max(); }
    SECTION("missing sample axis on a multidimensional input") {
        Tensor missing;
        REQUIRE(missing.create(DeviceType::CPU, DataType::F32, {2, 32}) == Result::SUCCESS);
        input = missing;
    }
    SECTION("duplicate axes") {
        REQUIRE(input.setAttribute("channelAxis", Index{0}) == Result::SUCCESS);
    }
    SECTION("out-of-range axis") {
        REQUIRE(input.setAttribute("sampleAxis", Index{1}) == Result::SUCCESS);
    }
    SECTION("incorrect axis type") {
        REQUIRE(input.setAttribute("sampleAxis", I64{0}) == Result::SUCCESS);
    }
    SECTION("unsupported dtype") {
        Tensor unsupported;
        REQUIRE(unsupported.create(DeviceType::CPU, DataType::U8, {64}) == Result::SUCCESS);
        REQUIRE(unsupported.setAttribute("sampleAxis", Index{0}) == Result::SUCCESS);
        input = unsupported;
    }
    SECTION("incorrect sample-rate type") {
        REQUIRE(input.setAttribute("sampleRate", F64{48000}) == Result::SUCCESS);
    }
    SECTION("nonfinite sample rate") {
        REQUIRE(input.setAttribute("sampleRate", std::numeric_limits<F32>::infinity()) ==
                Result::SUCCESS);
    }
    SECTION("unrepresentable output rate") {
        config.interpolation = 2;
        REQUIRE(input.setAttribute("sampleRate", std::numeric_limits<F32>::max()) ==
                Result::SUCCESS);
    }
    RequireValidationError(config, input);
}

TEST_CASE("Rational Resampler - Streaming preserves samples across layouts and skips",
          "[modules][rational_resampler][streaming]") {
    const std::array<std::pair<U64, U64>, 7> ratios{{
        {2, 3}, {3, 2}, {24, 125}, {5, 1}, {1, 5}, {1, 1}, {6, 4},
    }};
    for (const auto& [l, m] : ratios) {
        for (const U64 samples : {1, 7, 128}) {
            for (const std::string layout : {"contiguous", "sample-first", "strided"}) {
                DYNAMIC_SECTION("Ratio " << l << '/' << m << " samples " << samples
                                << " layout " << layout) {
                    Modules::RationalResampler config;
                    config.interpolation = l;
                    config.decimation = m;
                    RequireStreaming<F32>(config, samples, layout);
                    RequireStreaming<CF32>(config, samples, layout);
                }
            }
        }
    }
}

TEST_CASE("Rational Resampler - Matches independent upfirdn reference",
          "[modules][rational_resampler][numeric]") {
    const std::array<F32, 12> samples{1, -2, 3, 0.5f, -1, 0, 2, 4, -3, 1, 0.25f, -2};
    // SciPy: upfirdn(L * firwin(13, float(np.float32(.8)) / max(L, M),
    //                          window='blackman'), samples, up=L, down=M).
    // Keep only outputs supported by the input, without flushing the FIR tail.
    const std::vector<F32> down{
        0, 0.05235397769f, 0.1315927671f, 0.5294614968f,
        0.7298139518f, -0.36514678f, 2.037003688f, 0.5943010294f,
    };
    const std::vector<F32> up{
        0, -0.006985437099f, 0.3335236491f, 0.728330827f,
        -1.069217878f, -0.4099449239f, 2.494891393f, 1.33125747f,
        -0.001785250327f, -0.8316304645f, -0.3402461171f, 0.635374569f,
        2.007593054f, 3.412915874f, 1.75386088f, -2.253858906f,
        -0.306592428f, 0.7976459667f,
    };
    for (const bool interpolate : {false, true}) {
        for (const DataType dtype : {DataType::F32, DataType::CF32}) {
            CAPTURE(interpolate, dtype);
            Modules::RationalResampler config;
            config.interpolation = interpolate ? 3 : 2;
            config.decimation = interpolate ? 2 : 3;
            config.taps = 13;
            config.cutoff = 0.8f;
            Tensor input;
            REQUIRE(input.create(DeviceType::CPU, dtype, {samples.size()}) == Result::SUCCESS);
            REQUIRE(input.setAttribute("sampleAxis", Index{0}) == Result::SUCCESS);
            for (U64 i = 0; i < samples.size(); ++i) {
                if (dtype == DataType::F32) {
                    input.at<F32>(i) = samples[i];
                } else {
                    input.at<CF32>(i) = {samples[i], -2.0f * samples[i]};
                }
            }
            TestContext ctx("rational_resampler", DeviceType::CPU,
                            RuntimeType::NATIVE, "generic");
            ctx.setConfig(config);
            ctx.setInput("buffer", input);
            REQUIRE(ctx.run() == Result::SUCCESS);
            const auto& expected = interpolate ? up : down;
            const auto& output = ctx.output("buffer");
            REQUIRE(output.shape() == Shape{expected.size()});
            for (U64 i = 0; i < expected.size(); ++i) {
                if (dtype == DataType::F32) {
                    REQUIRE_THAT(output.at<F32>(i),
                                 Catch::Matchers::WithinAbs(expected[i], 2.0e-6f));
                } else {
                    REQUIRE(std::abs(output.at<CF32>(i) -
                                     CF32(expected[i], -2.0f * expected[i])) < 5.0e-6f);
                }
            }
        }
    }
}
