#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

#include "flowgraph_fixture.hh"
#include "jetstream/domains/core/ones_tensor/block.hh"
#include "jetstream/domains/dsp/spectrum_engine/block.hh"
#include "jetstream/registry.hh"

using namespace Jetstream;

TEST_CASE("Spectrum engine declares conditional module requirements",
          "[modules][dsp][spectrum_engine][requirements]") {
    const std::vector<Registry::BlockModuleRequirement> expected = {
        {"window"},
        {"invert"},
        {"reshape"},
        {"multiply"},
        {"fft"},
        {"amplitude"},
        {"agc", true},
        {"range", true},
    };

    const auto registrations = Registry::ListAvailableBlocks("spectrum_engine");
    REQUIRE(registrations.size() == 1);
    REQUIRE(registrations.front().moduleRequirements == expected);
}

TEST_CASE("Spectrum engine targets require only unconditional modules",
          "[modules][dsp][spectrum_engine][requirements]") {
    const auto targets = Registry::ListAvailableBlockTargets("spectrum_engine");
    const auto cpu = std::find_if(
        targets.begin(), targets.end(), [](const auto& target) {
            return target.device == DeviceType::CPU &&
                   target.runtime == RuntimeType::NATIVE &&
                   target.provider == "generic";
        });
    REQUIRE(cpu != targets.end());

    const auto cudaWindow = Registry::ListAvailableModules("window",
                                                            DeviceType::CUDA,
                                                            RuntimeType::NATIVE,
                                                            "generic");
    if (!cudaWindow.empty()) {
        const auto cuda = std::find_if(
            targets.begin(), targets.end(), [](const auto& target) {
                return target.device == DeviceType::CUDA &&
                       target.runtime == RuntimeType::NATIVE &&
                       target.provider == "generic";
            });
        REQUIRE(cuda != targets.end());
    }
}

TEST_CASE_METHOD(FlowgraphFixture,
                 "Spectrum engine block creates and exposes scaled spectrum",
                 "[modules][dsp][spectrum_engine][block]") {
    Parser::Map sourceConfig;
    sourceConfig["signalDataType"] = std::string("CF32");
    sourceConfig["bufferSize"] = std::string("256");

    REQUIRE(flowgraph->blockCreate("src", "signal_generator", sourceConfig, {}) ==
            Result::SUCCESS);

    Parser::Map engineConfig;
    engineConfig["axis"] = std::string("0");
    engineConfig["enableScale"] = std::string("true");
    engineConfig["rangeMin"] = std::string("-100");
    engineConfig["rangeMax"] = std::string("0");

    TensorMap inputs;
    inputs["buffer"].requested("src", "signal");

    REQUIRE(flowgraph->blockCreate("spec", "spectrum_engine", engineConfig, inputs) ==
            Result::SUCCESS);
    REQUIRE(viewBlock("spec").state == Block::State::Created);

    const Tensor out = viewBlock("spec").outputs.at("buffer").tensor;
    REQUIRE(out.dtype() == DataType::F32);
    REQUIRE(out.shape(0) == 256);
}

TEST_CASE_METHOD(FlowgraphFixture,
                 "Spectrum engine block rejects axis out of bounds",
                 "[modules][dsp][spectrum_engine][block][validation]") {
    Parser::Map sourceConfig;
    sourceConfig["signalDataType"] = std::string("CF32");
    sourceConfig["bufferSize"] = std::string("128");

    REQUIRE(flowgraph->blockCreate("src", "signal_generator", sourceConfig, {}) ==
            Result::SUCCESS);

    Parser::Map engineConfig;
    engineConfig["axis"] = std::string("2");

    TensorMap inputs;
    inputs["buffer"].requested("src", "signal");

    REQUIRE(flowgraph->blockCreate("spec_bad", "spectrum_engine", engineConfig, inputs) ==
            Result::SUCCESS);
    REQUIRE(viewBlock("spec_bad").state == Block::State::Errored);
}

TEST_CASE_METHOD(FlowgraphFixture,
                 "Spectrum engine applies window and FFT along a non-last axis",
                 "[modules][dsp][spectrum_engine][block][axis]") {
    Blocks::OnesTensor source;
    source.shape = {4, 3};
    source.dataType = "CF32";
    REQUIRE(flowgraph->blockCreate("src", source, {}) == Result::SUCCESS);

    TensorMap inputs;
    inputs["buffer"].requested("src", "buffer");

    Blocks::SpectrumEngine config;
    config.axis = 0;
    REQUIRE(flowgraph->blockCreate("spec", config, inputs) == Result::SUCCESS);
    REQUIRE(viewBlock("spec").state == Block::State::Created);
    REQUIRE(flowgraph->compute() == Result::SUCCESS);

    const Tensor out = viewBlock("spec").outputs.at("buffer").tensor;
    REQUIRE(out.shape() == Shape{4, 3});
    REQUIRE(out.dtype() == DataType::F32);

    const F32 expectedPeak = 20.0f * std::log10((0.42f * 3.0f) / 4.0f);
    for (U64 column = 0; column < out.shape(1); ++column) {
        REQUIRE_THAT(out.at<F32>(2, column),
                     Catch::Matchers::WithinAbs(expectedPeak, 0.1f));
    }
}
