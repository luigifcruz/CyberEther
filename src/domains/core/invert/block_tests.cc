#include <catch2/catch_test_macros.hpp>

#include "jetstream/domains/core/invert/block.hh"
#include "jetstream/domains/dsp/window/block.hh"
#include "flowgraph_fixture.hh"

using namespace Jetstream;

TEST_CASE_METHOD(FlowgraphFixture, "Invert block creates with complex input",
                 "[modules][invert][block]") {
    Blocks::Window source;
    REQUIRE(flowgraph->blockCreate("invert_src", source, {}) == Result::SUCCESS);

    TensorMap inputs;
    inputs["signal"] = {"invert_src", "window", {}};

    Blocks::Invert config;
    REQUIRE(flowgraph->blockCreate("invert_block", config, inputs) == Result::SUCCESS);
    REQUIRE(flowgraph->blockList().at("invert_block")->state() == Block::State::Created);
    REQUIRE(flowgraph->blockList().at("invert_block")->outputs().count("signal") == 1);
}

TEST_CASE_METHOD(FlowgraphFixture, "Invert block input reconnect lifecycle",
                 "[modules][invert][block][lifecycle]") {
    Blocks::Window source;
    REQUIRE(flowgraph->blockCreate("invert_life_src", source, {}) == Result::SUCCESS);

    TensorMap inputs;
    inputs["signal"] = {"invert_life_src", "window", {}};
    REQUIRE(flowgraph->blockCreate("invert_life", "invert", {}, inputs) ==
            Result::SUCCESS);

    REQUIRE(flowgraph->blockDisconnect("invert_life", "signal") == Result::SUCCESS);
    REQUIRE(flowgraph->blockList().at("invert_life")->state() == Block::State::Incomplete);

    REQUIRE(flowgraph->blockConnect("invert_life", "signal", "invert_life_src", "window")
            == Result::SUCCESS);
    REQUIRE(flowgraph->blockList().at("invert_life")->state() == Block::State::Created);
}
