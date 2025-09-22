#include <catch2/catch_test_macros.hpp>

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
    inputs["signal"] = {"src", "signal", {}};

    Blocks::Lineplot lineplotConfig;
    lineplotConfig.averaging = 2;
    lineplotConfig.decimation = 2;

    REQUIRE(flowgraph->blockCreate("lineplot", lineplotConfig, inputs) ==
            Result::SUCCESS);
    REQUIRE(flowgraph->blockList().at("lineplot")->state() == Block::State::Created);
    REQUIRE(flowgraph->blockList().at("lineplot")->outputs().empty());

    auto result = flowgraph->blockDisconnect("lineplot", "signal");
    REQUIRE((result == Result::SUCCESS || result == Result::INCOMPLETE));
    REQUIRE(flowgraph->blockList().at("lineplot")->state() ==
            Block::State::Incomplete);

    REQUIRE(flowgraph->blockConnect("lineplot", "signal", "src", "signal") ==
            Result::SUCCESS);
    REQUIRE(flowgraph->blockList().at("lineplot")->state() == Block::State::Created);
}

TEST_CASE_METHOD(FlowgraphFixture,
                 "Lineplot block reconfigure and validation",
                 "[modules][lineplot][block][validation]") {
    Blocks::SignalGenerator sourceConfig;
    sourceConfig.signalDataType = "F32";
    sourceConfig.bufferSize = 128;

    REQUIRE(flowgraph->blockCreate("src", sourceConfig, {}) == Result::SUCCESS);

    TensorMap inputs;
    inputs["signal"] = {"src", "signal", {}};

    REQUIRE(flowgraph->blockCreate("lineplot", Blocks::Lineplot(), inputs) ==
            Result::SUCCESS);

    Parser::Map config;
    config["averaging"] = std::string("8");
    config["decimation"] = std::string("2");
    REQUIRE(flowgraph->blockReconfigure("lineplot", config) == Result::SUCCESS);
    REQUIRE(flowgraph->blockList().at("lineplot")->state() == Block::State::Created);

    Blocks::Lineplot invalid;
    invalid.averaging = 0;
    REQUIRE(flowgraph->blockCreate("lineplot_invalid", invalid, inputs) ==
            Result::SUCCESS);
    REQUIRE(flowgraph->blockList().at("lineplot_invalid")->state() ==
            Block::State::Errored);
}
