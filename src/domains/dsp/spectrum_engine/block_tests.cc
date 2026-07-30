#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <algorithm>
#include <cmath>
#include <string>
#include <unordered_map>
#include <vector>

#include "flowgraph_fixture.hh"
#include "jetstream/domains/core/ones_tensor/block.hh"
#include "jetstream/domains/dsp/spectrum_engine/block.hh"
#include "jetstream/registry.hh"

using namespace Jetstream;

TEST_CASE("Spectrum engine axis defaults to the last dimension",
          "[modules][dsp][spectrum_engine][config]") {
    REQUIRE(Blocks::SpectrumEngine{}.axis == -1);
}

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
    const auto block = viewBlock("spec");
    REQUIRE(block.state == Block::State::Created);

    const auto axis = std::find_if(block.interfaceConfigs.begin(),
                                   block.interfaceConfigs.end(),
                                   [](const auto& entry) { return entry.name == "axis"; });
    REQUIRE(axis != block.interfaceConfigs.end());
    REQUIRE(axis->format == "int:");

    Parser::Map saved;
    REQUIRE(flowgraph->blockConfig("spec", saved) == Result::SUCCESS);
    REQUIRE(saved.at("axis").type() == typeid(I64));
    REQUIRE(std::any_cast<I64>(saved.at("axis")) == 0);

    const Tensor out = block.outputs.at("buffer").tensor;
    REQUIRE(out.dtype() == DataType::F32);
    REQUIRE(out.shape(0) == 256);

    for (U64 cycle = 0; cycle < 3; ++cycle) {
        REQUIRE(flowgraph->compute() == Result::SUCCESS);
    }

    std::vector<Flowgraph::View::MetricEntry> sourceMetrics;
    REQUIRE(flowgraph->view().metrics("src", sourceMetrics) == Result::SUCCESS);
    REQUIRE(sourceMetrics.size() == 1);
    const auto* sourceTiming = std::any_cast<Module::Timing>(&sourceMetrics.front().value);
    REQUIRE(sourceTiming != nullptr);
    REQUIRE(sourceTiming->cycles == 3);

    std::vector<Flowgraph::View::MetricEntry> engineMetrics;
    REQUIRE(flowgraph->view().metrics("spec", engineMetrics) == Result::SUCCESS);

    std::unordered_map<std::string, U64> cycles;
    std::unordered_map<std::string, F32> computeTimes;
    for (const auto& metric : engineMetrics) {
        const auto* timing = std::any_cast<Module::Timing>(&metric.value);
        REQUIRE(timing != nullptr);
        cycles[metric.name] = timing->cycles;
        computeTimes[metric.name] = timing->computeTime;
    }

    REQUIRE(cycles.at("runtime:window") == 1);
    REQUIRE(cycles.at("runtime:invert") == 1);
    REQUIRE(cycles.at("runtime:multiply") == 3);
    REQUIRE(cycles.at("runtime:fft") == 3);
    REQUIRE(computeTimes.at("runtime:window") == 0.0f);
    REQUIRE(computeTimes.at("runtime:invert") == 0.0f);
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

    engineConfig["axis"] = std::string("-2");
    REQUIRE(flowgraph->blockCreate("spec_too_negative", "spectrum_engine", engineConfig, inputs) ==
            Result::SUCCESS);
    REQUIRE(viewBlock("spec_too_negative").state == Block::State::Errored);
}

TEST_CASE_METHOD(FlowgraphFixture,
                 "Spectrum engine defers axis validation for incomplete input",
                 "[modules][dsp][spectrum_engine][block][validation]") {
    Blocks::SpectrumEngine config;
    config.axis = 2;

    SECTION("absent input") {
        REQUIRE(flowgraph->blockCreate("spec_absent", config, {}) == Result::SUCCESS);
        REQUIRE(viewBlock("spec_absent").state == Block::State::Incomplete);
        REQUIRE(viewBlock("spec_absent").outputs.empty());
    }

    SECTION("unresolved input") {
        Blocks::OnesTensor source;
        source.shape = {4};
        source.dataType = "CF32";
        REQUIRE(flowgraph->blockCreate("spec_unresolved_src", source, {}) ==
                Result::SUCCESS);

        TensorMap inputs;
        inputs["buffer"].requested("spec_unresolved_src", "missing");

        REQUIRE(flowgraph->blockCreate("spec_unresolved", config, inputs) ==
                Result::SUCCESS);
        REQUIRE(viewBlock("spec_unresolved").state == Block::State::Incomplete);
        REQUIRE(viewBlock("spec_unresolved").outputs.empty());
    }
}

TEST_CASE_METHOD(FlowgraphFixture,
                  "Spectrum engine preserves invalid axis for recovery",
                 "[modules][dsp][spectrum_engine][block][reconfigure][validation]") {
    Blocks::OnesTensor source;
    source.shape = {4, 3};
    source.dataType = "CF32";
    REQUIRE(flowgraph->blockCreate("spec_update_src", source, {}) == Result::SUCCESS);

    TensorMap inputs;
    inputs["buffer"].requested("spec_update_src", "buffer");

    Blocks::SpectrumEngine config;
    config.axis = 1;
    config.rangeMin = -90.0f;
    config.rangeMax = -5.0f;
    REQUIRE(flowgraph->blockCreate("spec_update", config, inputs) == Result::SUCCESS);
    REQUIRE(flowgraph->compute() == Result::SUCCESS);

    Parser::Map update;
    update["axis"] = I64{2};
    update["enableScale"] = true;
    REQUIRE(flowgraph->blockReconfigure("spec_update", update) == Result::SUCCESS);

    const auto errored = viewBlock("spec_update");
    REQUIRE(errored.state == Block::State::Errored);
    REQUIRE(errored.outputs.empty());
    REQUIRE(errored.interfaceInputs.size() == 1);
    REQUIRE(errored.interfaceOutputs.size() == 1);
    REQUIRE(errored.interfaceConfigs.size() == 5);

    Parser::Map savedMap;
    REQUIRE(flowgraph->blockConfig("spec_update", savedMap) == Result::SUCCESS);
    Blocks::SpectrumEngine saved;
    REQUIRE(saved.deserialize(savedMap) == Result::SUCCESS);
    REQUIRE(saved.axis == 2);
    REQUIRE(saved.enableAgc == config.enableAgc);
    REQUIRE(saved.enableScale);
    REQUIRE(saved.rangeMin == config.rangeMin);
    REQUIRE(saved.rangeMax == config.rangeMax);

    Parser::Map recovery;
    recovery["axis"] = I64{1};
    REQUIRE(flowgraph->blockReconfigure("spec_update", recovery) == Result::SUCCESS);
    const auto recovered = viewBlock("spec_update");
    REQUIRE(recovered.state == Block::State::Created);
    Tensor output = recovered.outputs.at("buffer").tensor;
    std::fill(output.data<F32>(), output.data<F32>() + output.size(), 12345.0f);
    REQUIRE(flowgraph->compute() == Result::SUCCESS);
    for (U64 index = 0; index < output.size(); ++index) {
        REQUIRE(output.data<F32>()[index] != 12345.0f);
    }
}

TEST_CASE_METHOD(FlowgraphFixture,
                 "Spectrum engine recreates for valid axis and topology changes",
                 "[modules][dsp][spectrum_engine][block][reconfigure]") {
    Blocks::OnesTensor source;
    source.shape = {4, 3};
    source.dataType = "CF32";
    REQUIRE(flowgraph->blockCreate("spec_recreate_src", source, {}) == Result::SUCCESS);

    TensorMap inputs;
    inputs["buffer"].requested("spec_recreate_src", "buffer");

    Blocks::SpectrumEngine config;
    config.axis = 1;
    REQUIRE(flowgraph->blockCreate("spec_recreate", config, inputs) == Result::SUCCESS);
    const auto originalOutputId =
        viewBlock("spec_recreate").outputs.at("buffer").tensor.id();

    Parser::Map update;
    update["axis"] = I64{0};
    update["enableScale"] = true;
    update["rangeMin"] = -100.0f;
    update["rangeMax"] = -10.0f;
    REQUIRE(flowgraph->blockReconfigure("spec_recreate", update) == Result::SUCCESS);

    const auto recreated = viewBlock("spec_recreate");
    REQUIRE(recreated.state == Block::State::Created);
    REQUIRE(recreated.outputs.at("buffer").tensor.id() != originalOutputId);
    REQUIRE(recreated.outputs.at("buffer").tensor.shape() == Shape{4, 3});
    REQUIRE(recreated.outputs.at("buffer").tensor.dtype() == DataType::F32);

    Parser::Map savedMap;
    REQUIRE(flowgraph->blockConfig("spec_recreate", savedMap) == Result::SUCCESS);
    Blocks::SpectrumEngine saved;
    REQUIRE(saved.deserialize(savedMap) == Result::SUCCESS);
    REQUIRE(saved.axis == 0);
    REQUIRE(saved.enableScale);
    REQUIRE(saved.rangeMin == -100.0f);
    REQUIRE(saved.rangeMax == -10.0f);

    Tensor output = recreated.outputs.at("buffer").tensor;
    std::fill(output.data<F32>(), output.data<F32>() + output.size(), 12345.0f);
    REQUIRE(flowgraph->compute() == Result::SUCCESS);
    REQUIRE(std::any_of(output.data<F32>(), output.data<F32>() + output.size(),
                        [](const F32 value) { return value != 12345.0f; }));
}

TEST_CASE_METHOD(FlowgraphFixture,
                 "Spectrum engine delegates dtype validation to child modules",
                 "[modules][dsp][spectrum_engine][block][validation]") {
    Blocks::OnesTensor source;
    source.shape = {4};
    source.dataType = "F64";
    REQUIRE(flowgraph->blockCreate("spec_dtype_src", source, {}) == Result::SUCCESS);

    TensorMap inputs;
    inputs["buffer"].requested("spec_dtype_src", "buffer");

    Blocks::SpectrumEngine config;
    config.axis = 0;
    config.enableScale = true;
    REQUIRE(flowgraph->blockCreate("spec_dtype", config, inputs) == Result::SUCCESS);
    const auto block = viewBlock("spec_dtype");
    REQUIRE(block.state == Block::State::Errored);
    REQUIRE(block.outputs.empty());
    REQUIRE(std::any_of(block.interfaceConfigs.begin(),
                        block.interfaceConfigs.end(),
                        [](const auto& field) { return field.name == "rangeMin"; }));
    REQUIRE(std::any_of(block.interfaceConfigs.begin(),
                        block.interfaceConfigs.end(),
                        [](const auto& field) { return field.name == "rangeMax"; }));
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

    config.axis = -2;
    REQUIRE(flowgraph->blockCreate("spec_negative", config, inputs) == Result::SUCCESS);
    REQUIRE(viewBlock("spec_negative").state == Block::State::Created);
    REQUIRE(flowgraph->compute() == Result::SUCCESS);

    const Tensor positiveOut = viewBlock("spec").outputs.at("buffer").tensor;
    const Tensor negativeOut = viewBlock("spec_negative").outputs.at("buffer").tensor;
    REQUIRE(positiveOut.shape() == Shape{4, 3});
    REQUIRE(negativeOut.shape() == positiveOut.shape());
    REQUIRE(positiveOut.dtype() == DataType::F32);
    REQUIRE(negativeOut.dtype() == positiveOut.dtype());

    const F32 expectedPeak = 20.0f * std::log10((0.42f * 3.0f) / 4.0f);
    for (U64 column = 0; column < positiveOut.shape(1); ++column) {
        REQUIRE_THAT(positiveOut.at<F32>(2, column),
                     Catch::Matchers::WithinAbs(expectedPeak, 0.1f));
        REQUIRE_THAT(negativeOut.at<F32>(2, column),
                     Catch::Matchers::WithinAbs(expectedPeak, 0.1f));
        REQUIRE_THAT(negativeOut.at<F32>(2, column),
                     Catch::Matchers::WithinAbs(positiveOut.at<F32>(2, column), 1e-4f));
    }
}
