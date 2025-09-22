#include <catch2/catch_test_macros.hpp>

#include <string>

#include "jetstream/domains/dsp/signal_generator/block.hh"
#include "jetstream/domains/visualization/constellation/block.hh"
#include "flowgraph_fixture.hh"

using namespace Jetstream;

TEST_CASE_METHOD(FlowgraphFixture,
                 "Constellation block wiring and lifecycle",
                 "[modules][constellation][block]") {
    Blocks::SignalGenerator sourceConfig;
    sourceConfig.signalDataType = "CF32";
    sourceConfig.bufferSize = 64;

    REQUIRE(flowgraph->blockCreate("src", sourceConfig, {}) == Result::SUCCESS);

    TensorMap inputs;
    inputs["signal"] = {"src", "signal", {}};

    Blocks::Constellation constellationConfig;
    REQUIRE(flowgraph->blockCreate("constellation", constellationConfig, inputs) ==
            Result::SUCCESS);
    REQUIRE(flowgraph->blockList().at("constellation")->state() ==
            Block::State::Created);
    REQUIRE(flowgraph->blockList().at("constellation")->outputs().empty());

    auto result = flowgraph->blockDisconnect("constellation", "signal");
    REQUIRE((result == Result::SUCCESS || result == Result::INCOMPLETE));
    REQUIRE(flowgraph->blockList().at("constellation")->state() ==
            Block::State::Incomplete);

    REQUIRE(flowgraph->blockConnect("constellation", "signal", "src", "signal") ==
            Result::SUCCESS);
    REQUIRE(flowgraph->blockList().at("constellation")->state() ==
            Block::State::Created);
}
