#include <catch2/catch_test_macros.hpp>

#include <any>
#include <string>

#include "jetstream/domains/dsp/signal_generator/block.hh"
#include "jetstream/domains/visualization/lineplot/block.hh"
#include "flowgraph_fixture.hh"

using namespace Jetstream;

TEST_CASE_METHOD(FlowgraphFixture,
                 "Lineplot block create and lifecycle",
                 "[modules][lineplot][block]") {
    Blocks::SignalGenerator sourceConfig;
    sourceConfig.signalDataType = "F32";
    sourceConfig.bufferSize = 128;

    REQUIRE(flowgraph->blockCreate("src", sourceConfig, {}) == Result::SUCCESS);

    TensorMap inputs;
    inputs["signal"].requested("src", "signal");

    Blocks::Lineplot lineplotConfig;
    lineplotConfig.averaging = 2;
    lineplotConfig.decimation = 2;
    lineplotConfig.fill = false;
    lineplotConfig.rangeMin = -1.0f;
    lineplotConfig.rangeMax = 1.0f;
    lineplotConfig.xLabel = "Time";
    lineplotConfig.yLabel = "Voltage";

    REQUIRE(flowgraph->blockCreate("lineplot", lineplotConfig, inputs) ==
            Result::SUCCESS);
    const auto block = viewBlock("lineplot");
    REQUIRE(block.state == Block::State::Created);
    REQUIRE(block.outputs.empty());
    REQUIRE_FALSE(std::any_cast<bool>(block.config.at("fill")));
    REQUIRE(std::any_cast<F32>(block.config.at("rangeMin")) == -1.0f);
    REQUIRE(std::any_cast<F32>(block.config.at("rangeMax")) == 1.0f);
    REQUIRE(std::any_cast<std::string>(block.config.at("xLabel")) == "Time");
    REQUIRE(std::any_cast<std::string>(block.config.at("yLabel")) == "Voltage");
    REQUIRE_FALSE(block.config.contains("thickness"));
    for (const auto& entry : block.interfaceConfigs) {
        REQUIRE(entry.name != "fill");
        REQUIRE(entry.name != "rangeMin");
        REQUIRE(entry.name != "rangeMax");
        REQUIRE(entry.name != "xLabel");
        REQUIRE(entry.name != "yLabel");
    }

    auto result = flowgraph->blockDisconnect("lineplot", "signal");
    REQUIRE((result == Result::SUCCESS || result == Result::INCOMPLETE));
    REQUIRE(viewBlock("lineplot").state ==
            Block::State::Incomplete);

    REQUIRE(flowgraph->blockConnect("lineplot", "signal", "src", "signal") ==
            Result::SUCCESS);
    REQUIRE(viewBlock("lineplot").state == Block::State::Created);
}

TEST_CASE_METHOD(FlowgraphFixture,
                 "Lineplot block reconfigure and validation",
                 "[modules][lineplot][block][validation]") {
    Blocks::SignalGenerator sourceConfig;
    sourceConfig.signalDataType = "F32";
    sourceConfig.bufferSize = 128;

    REQUIRE(flowgraph->blockCreate("src", sourceConfig, {}) == Result::SUCCESS);

    TensorMap inputs;
    inputs["signal"].requested("src", "signal");

    REQUIRE(flowgraph->blockCreate("lineplot", Blocks::Lineplot(), inputs) ==
            Result::SUCCESS);
    const auto initial = viewBlock("lineplot");
    REQUIRE(std::any_cast<std::string>(initial.config.at("xLabel")) ==
            "Frequency (MHz)");
    REQUIRE(std::any_cast<std::string>(initial.config.at("yLabel")) ==
            "Amplitude (dBFS)");
    REQUIRE(std::any_cast<F32>(initial.config.at("rangeMin")) == -100.0f);
    REQUIRE(std::any_cast<F32>(initial.config.at("rangeMax")) == 0.0f);

    Parser::Map config;
    config["averaging"] = std::string("8");
    config["decimation"] = std::string("2");
    config["rangeMin"] = std::string("-1");
    config["rangeMax"] = std::string("1");
    config["xLabel"] = std::string();
    config["yLabel"] = std::string("Amplitude");
    REQUIRE(flowgraph->blockReconfigure("lineplot", config) == Result::SUCCESS);
    const auto reconfigured = viewBlock("lineplot");
    REQUIRE(reconfigured.state == Block::State::Created);
    REQUIRE(std::any_cast<std::string>(reconfigured.config.at("xLabel")).empty());
    REQUIRE(std::any_cast<std::string>(reconfigured.config.at("yLabel")) ==
            "Amplitude");
    REQUIRE(std::any_cast<F32>(reconfigured.config.at("rangeMin")) == -1.0f);
    REQUIRE(std::any_cast<F32>(reconfigured.config.at("rangeMax")) == 1.0f);

    Blocks::Lineplot invalid;
    invalid.averaging = 0;
    REQUIRE(flowgraph->blockCreate("lineplot_invalid", invalid, inputs) ==
            Result::SUCCESS);
    REQUIRE(viewBlock("lineplot_invalid").state ==
            Block::State::Errored);

    Blocks::Lineplot invalidRange;
    invalidRange.rangeMin = 1.0f;
    invalidRange.rangeMax = 1.0f;
    REQUIRE(flowgraph->blockCreate("lineplot_invalid_range", invalidRange,
                                   inputs) == Result::SUCCESS);
    REQUIRE(viewBlock("lineplot_invalid_range").state ==
            Block::State::Errored);
}

TEST_CASE_METHOD(FlowgraphFixture,
                 "Lineplot block delegates dtype validation to its module",
                 "[modules][lineplot][block][validation]") {
    Blocks::SignalGenerator source;
    source.bufferSize = 64;
    source.signalDataType = "CF32";
    REQUIRE(flowgraph->blockCreate("lineplot_dtype_src", source, {}) ==
            Result::SUCCESS);

    TensorMap inputs;
    inputs["signal"].requested("lineplot_dtype_src", "signal");
    REQUIRE(flowgraph->blockCreate("lineplot_dtype", Blocks::Lineplot{}, inputs) ==
            Result::SUCCESS);

    const auto block = viewBlock("lineplot_dtype");
    REQUIRE(block.state == Block::State::Errored);
    REQUIRE(block.outputs.empty());
}

TEST_CASE_METHOD(FlowgraphFixture,
                  "Lineplot block preserves a rejected module update for recovery",
                 "[modules][lineplot][block][reconfigure][validation]") {
    Blocks::SignalGenerator source;
    source.signalDataType = "F32";
    source.bufferSize = 64;
    REQUIRE(flowgraph->blockCreate("lineplot_update_src", source, {}) ==
            Result::SUCCESS);

    Blocks::Lineplot config;
    config.averaging = 2;
    TensorMap inputs;
    inputs["signal"].requested("lineplot_update_src", "signal");
    REQUIRE(flowgraph->blockCreate("lineplot_update", config, inputs) ==
            Result::SUCCESS);
    REQUIRE(flowgraph->compute() == Result::SUCCESS);

    Parser::Map invalidUpdate;
    invalidUpdate["averaging"] = U64{0};
    REQUIRE(flowgraph->blockReconfigure("lineplot_update", invalidUpdate) ==
            Result::SUCCESS);
    REQUIRE(viewBlock("lineplot_update").state == Block::State::Errored);

    Parser::Map validSparseUpdate;
    validSparseUpdate["averaging"] = U64{8};
    REQUIRE(flowgraph->blockReconfigure("lineplot_update", validSparseUpdate) ==
            Result::SUCCESS);

    Parser::Map saved;
    REQUIRE(flowgraph->blockConfig("lineplot_update", saved) == Result::SUCCESS);
    REQUIRE(std::any_cast<U64>(saved.at("averaging")) == 8);
    REQUIRE(flowgraph->compute() == Result::SUCCESS);
}
