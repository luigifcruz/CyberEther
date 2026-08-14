#include <catch2/catch_test_macros.hpp>

#include <any>
#include <algorithm>
#include <string>
#include <vector>

#include "flowgraph_fixture.hh"
#include "jetstream/domains/core/ones_tensor/block.hh"
#include "jetstream/domains/dsp/signal_generator/block.hh"
#include "jetstream/domains/visualization/spectrum_analyzer/block.hh"
#include "jetstream/memory/axis.hh"
#include "jetstream/registry.hh"

using namespace Jetstream;

TEST_CASE("Spectrum Analyzer derives its transform from signal metadata",
          "[modules][spectrum-analyzer][config]") {
    Blocks::SpectrumAnalyzer config;

    Parser::Map serialized;
    REQUIRE(config.serialize(serialized) == Result::SUCCESS);
    REQUIRE_FALSE(serialized.contains("axis"));
    REQUIRE_FALSE(serialized.contains("thickness"));
}

TEST_CASE("Spectrum Analyzer declares its complete module chain",
          "[modules][spectrum-analyzer][requirements]") {
    const std::vector<Registry::BlockModuleRequirement> expected = {
        {"window"},
        {"invert"},
        {"reshape"},
        {"multiply"},
        {"fft"},
        {"amplitude"},
        {"range"},
        {"signal_view"},
    };

    const auto registrations = Registry::ListAvailableBlocks("spectrum_analyzer");
    REQUIRE(registrations.size() == 1);
    REQUIRE(registrations.front().moduleRequirements == expected);
}

TEST_CASE_METHOD(FlowgraphFixture,
                 "Spectrum Analyzer block creates one combined surface",
                 "[modules][spectrum-analyzer][block]") {
    Blocks::SignalGenerator sourceConfig;
    sourceConfig.signalDataType = "CF32";
    sourceConfig.bufferSize = 128;
    REQUIRE(flowgraph->blockCreate("src", sourceConfig, {}) == Result::SUCCESS);

    TensorMap inputs;
    inputs["buffer"].requested("src", "signal");

    Blocks::SpectrumAnalyzer config;
    config.waterfallHeight = 32;
    config.fill = false;
    config.xLabel = "Frequency";
    config.amplitudeLabel = "Power";
    config.waterfallLabel = "History";
    REQUIRE(flowgraph->blockCreate("analyzer", config, inputs) == Result::SUCCESS);
    const auto block = viewBlock("analyzer");
    REQUIRE(block.state == Block::State::Created);
    REQUIRE(block.outputs.empty());
    REQUIRE(block.surfaces.size() == 1);
    REQUIRE_FALSE(std::any_cast<bool>(block.config.at("fill")));
    REQUIRE(std::any_cast<std::string>(block.config.at("xLabel")) == "Frequency");
    REQUIRE(std::any_cast<std::string>(block.config.at("amplitudeLabel")) == "Power");
    REQUIRE(std::any_cast<std::string>(block.config.at("waterfallLabel")) == "History");
    for (const auto& entry : block.interfaceConfigs) {
        REQUIRE(entry.name != "axis");
        REQUIRE(entry.name != "fill");
        REQUIRE(entry.name != "xLabel");
        REQUIRE(entry.name != "amplitudeLabel");
        REQUIRE(entry.name != "waterfallLabel");
    }
    REQUIRE(flowgraph->compute() == Result::SUCCESS);

    Parser::Map update;
    update["rangeMin"] = std::string("-100");
    update["averaging"] = std::string("2");
    update["xLabel"] = std::string("Offset");
    update["amplitudeLabel"] = std::string("Level");
    update["waterfallLabel"] = std::string("Time");
    REQUIRE(flowgraph->blockReconfigure("analyzer", update) == Result::SUCCESS);
    const auto reconfigured = viewBlock("analyzer");
    REQUIRE(std::any_cast<std::string>(reconfigured.config.at("xLabel")) == "Offset");
    REQUIRE(std::any_cast<std::string>(reconfigured.config.at("amplitudeLabel")) ==
            "Level");
    REQUIRE(std::any_cast<std::string>(reconfigured.config.at("waterfallLabel")) ==
            "Time");
    REQUIRE(flowgraph->compute() == Result::SUCCESS);
}

TEST_CASE_METHOD(FlowgraphFixture,
                 "Spectrum Analyzer validates transform and plot settings",
                 "[modules][spectrum-analyzer][block][validation]") {
    Blocks::SignalGenerator sourceConfig;
    sourceConfig.signalDataType = "CF32";
    sourceConfig.bufferSize = 128;
    REQUIRE(flowgraph->blockCreate("src", sourceConfig, {}) == Result::SUCCESS);

    TensorMap inputs;
    inputs["buffer"].requested("src", "signal");

    Blocks::SpectrumAnalyzer invalidHeight;
    invalidHeight.waterfallHeight = 0;
    REQUIRE(flowgraph->blockCreate("invalid_height", invalidHeight, inputs) ==
            Result::SUCCESS);
    REQUIRE(viewBlock("invalid_height").state == Block::State::Errored);

    Blocks::SignalGenerator realSourceConfig;
    realSourceConfig.signalDataType = "F32";
    realSourceConfig.bufferSize = 128;
    REQUIRE(flowgraph->blockCreate("real_src", realSourceConfig, {}) ==
            Result::SUCCESS);

    TensorMap realInputs;
    realInputs["buffer"].requested("real_src", "signal");
    REQUIRE(flowgraph->blockCreate("invalid_dtype",
                                   Blocks::SpectrumAnalyzer{},
                                   realInputs) == Result::SUCCESS);
    REQUIRE(viewBlock("invalid_dtype").state == Block::State::Errored);
}

TEST_CASE_METHOD(FlowgraphFixture,
                 "Spectrum Analyzer supports leading and trailing batch layouts",
                 "[modules][spectrum-analyzer][block][metadata]") {
    Blocks::OnesTensor source;
    source.dataType = "CF32";
    Index sampleAxis = 1;
    Index batchAxis = 0;

    SECTION("leading batch") {
        source.shape = {2, 64};
    }
    SECTION("trailing batch") {
        source.shape = {64, 2};
        sampleAxis = 0;
        batchAxis = 1;
    }

    REQUIRE(flowgraph->blockCreate("src", source, {}) == Result::SUCCESS);
    Tensor sourceTensor = viewBlock("src").outputs.at("buffer").tensor;
    REQUIRE(SetSignalAxes(sourceTensor, {
        .sample = sampleAxis,
        .batch = batchAxis,
    }) == Result::SUCCESS);

    TensorMap inputs;
    inputs["buffer"].requested("src", "buffer");

    Blocks::SpectrumAnalyzer config;
    config.waterfallHeight = 32;
    REQUIRE(flowgraph->blockCreate("analyzer", config, inputs) == Result::SUCCESS);
    REQUIRE(viewBlock("analyzer").state == Block::State::Created);
    REQUIRE(flowgraph->compute() == Result::SUCCESS);
}

TEST_CASE_METHOD(FlowgraphFixture,
                 "Spectrum Analyzer rejects missing signal metadata",
                 "[modules][spectrum-analyzer][block][validation][metadata]") {
    Blocks::OnesTensor source;
    source.shape = {2, 64};
    source.dataType = "CF32";
    REQUIRE(flowgraph->blockCreate("src", source, {}) == Result::SUCCESS);

    TensorMap inputs;
    inputs["buffer"].requested("src", "buffer");
    REQUIRE(flowgraph->blockCreate("analyzer", Blocks::SpectrumAnalyzer{}, inputs) ==
            Result::SUCCESS);
    REQUIRE(viewBlock("analyzer").state == Block::State::Errored);
}

TEST_CASE_METHOD(FlowgraphFixture,
                 "Spectrum Analyzer preserves invalid config for recovery",
                 "[modules][spectrum-analyzer][block][reconfigure][validation]") {
    Blocks::SignalGenerator source;
    source.signalDataType = "CF32";
    source.bufferSize = 64;
    REQUIRE(flowgraph->blockCreate("src", source, {}) == Result::SUCCESS);

    TensorMap inputs;
    inputs["buffer"].requested("src", "signal");
    Blocks::SpectrumAnalyzer config;
    config.waterfallHeight = 32;
    REQUIRE(flowgraph->blockCreate("analyzer", config, inputs) == Result::SUCCESS);

    Parser::Map update;
    update["waterfallHeight"] = U64{0};
    REQUIRE(flowgraph->blockReconfigure("analyzer", update) == Result::SUCCESS);
    REQUIRE(viewBlock("analyzer").state == Block::State::Errored);

    Parser::Map saved;
    REQUIRE(flowgraph->blockConfig("analyzer", saved) == Result::SUCCESS);
    REQUIRE(std::any_cast<U64>(saved.at("waterfallHeight")) == 0);

    Parser::Map recovery;
    recovery["waterfallHeight"] = config.waterfallHeight;
    REQUIRE(flowgraph->blockReconfigure("analyzer", recovery) == Result::SUCCESS);
    REQUIRE(viewBlock("analyzer").state == Block::State::Created);
}
