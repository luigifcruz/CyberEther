#include <catch2/catch_test_macros.hpp>

#include <string>

#include "jetstream/domains/dsp/signal_generator/block.hh"
#include "jetstream/domains/visualization/waterfall/block.hh"
#include "flowgraph_fixture.hh"

using namespace Jetstream;

TEST_CASE_METHOD(FlowgraphFixture,
                 "Waterfall block create and lifecycle",
                 "[modules][waterfall][block]") {
    Blocks::SignalGenerator sourceConfig;
    sourceConfig.signalDataType = "F32";
    sourceConfig.bufferSize = 64;

    REQUIRE(flowgraph->blockCreate("src", sourceConfig, {}) == Result::SUCCESS);

    TensorMap inputs;
    inputs["signal"].requested("src", "signal");

    Blocks::Waterfall config;
    config.height = 64;
    config.interpolate = false;

    REQUIRE(flowgraph->blockCreate("waterfall", config, inputs) == Result::SUCCESS);
    REQUIRE(flowgraph->blockList().at("waterfall")->state() == Block::State::Created);
    REQUIRE(flowgraph->blockList().at("waterfall")->outputs().empty());

    auto result = flowgraph->blockDisconnect("waterfall", "signal");
    REQUIRE((result == Result::SUCCESS || result == Result::INCOMPLETE));
    REQUIRE(flowgraph->blockList().at("waterfall")->state() ==
            Block::State::Incomplete);

    REQUIRE(flowgraph->blockConnect("waterfall", "signal", "src", "signal") ==
            Result::SUCCESS);
    REQUIRE(flowgraph->blockList().at("waterfall")->state() == Block::State::Created);
}

TEST_CASE_METHOD(FlowgraphFixture,
                 "Waterfall block reconfigure and validation",
                 "[modules][waterfall][block][validation]") {
    Blocks::SignalGenerator sourceConfig;
    sourceConfig.signalDataType = "F32";
    sourceConfig.bufferSize = 64;

    REQUIRE(flowgraph->blockCreate("src", sourceConfig, {}) == Result::SUCCESS);

    TensorMap inputs;
    inputs["signal"].requested("src", "signal");

    REQUIRE(flowgraph->blockCreate("waterfall", Blocks::Waterfall(), inputs) ==
            Result::SUCCESS);

    Parser::Map config;
    config["height"] = std::string("128");
    config["interpolate"] = std::string("false");
    REQUIRE(flowgraph->blockReconfigure("waterfall", config) == Result::SUCCESS);
    REQUIRE(flowgraph->blockList().at("waterfall")->state() == Block::State::Created);

    Blocks::Waterfall invalid;
    invalid.height = 0;
    REQUIRE(flowgraph->blockCreate("waterfall_invalid", invalid, inputs) ==
            Result::SUCCESS);
    REQUIRE(flowgraph->blockList().at("waterfall_invalid")->state() ==
            Block::State::Errored);
}
