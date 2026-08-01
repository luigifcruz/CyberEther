#include <catch2/catch_test_macros.hpp>

#include <any>
#include <limits>
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

    REQUIRE(flowgraph->blockCreate("lineplot", lineplotConfig, inputs) ==
            Result::SUCCESS);
    REQUIRE(viewBlock("lineplot").state == Block::State::Created);
    REQUIRE(viewBlock("lineplot").outputs.empty());

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

    Parser::Map config;
    config["averaging"] = std::string("8");
    config["decimation"] = std::string("2");
    REQUIRE(flowgraph->blockReconfigure("lineplot", config) == Result::SUCCESS);
    REQUIRE(viewBlock("lineplot").state == Block::State::Created);

    Blocks::Lineplot invalid;
    invalid.averaging = 0;
    REQUIRE(flowgraph->blockCreate("lineplot_invalid", invalid, inputs) ==
            Result::SUCCESS);
    REQUIRE(viewBlock("lineplot_invalid").state ==
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
    config.thickness = 1.5f;
    TensorMap inputs;
    inputs["signal"].requested("lineplot_update_src", "signal");
    REQUIRE(flowgraph->blockCreate("lineplot_update", config, inputs) ==
            Result::SUCCESS);
    REQUIRE(flowgraph->compute() == Result::SUCCESS);

    Parser::Map invalidUpdate;
    invalidUpdate["thickness"] = std::numeric_limits<F32>::infinity();
    REQUIRE(flowgraph->blockReconfigure("lineplot_update", invalidUpdate) ==
            Result::SUCCESS);
    REQUIRE(viewBlock("lineplot_update").state == Block::State::Errored);

    Parser::Map validSparseUpdate;
    validSparseUpdate["averaging"] = U64{8};
    validSparseUpdate["thickness"] = config.thickness;
    REQUIRE(flowgraph->blockReconfigure("lineplot_update", validSparseUpdate) ==
            Result::SUCCESS);

    Parser::Map saved;
    REQUIRE(flowgraph->blockConfig("lineplot_update", saved) == Result::SUCCESS);
    REQUIRE(std::any_cast<U64>(saved.at("averaging")) == 8);
    REQUIRE(std::any_cast<F32>(saved.at("thickness")) == config.thickness);
    REQUIRE(flowgraph->compute() == Result::SUCCESS);
}
