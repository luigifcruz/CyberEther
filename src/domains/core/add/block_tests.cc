#include <catch2/catch_test_macros.hpp>

#include "jetstream/domains/core/add/block.hh"
#include "jetstream/domains/dsp/window/block.hh"
#include "flowgraph_fixture.hh"

using namespace Jetstream;

TEST_CASE_METHOD(FlowgraphFixture, "Add block creates and exposes output",
                 "[modules][add][block]") {
    Blocks::Window source;
    source.size = 16;
    REQUIRE(flowgraph->blockCreate("add_src_a", source, {}) == Result::SUCCESS);
    REQUIRE(flowgraph->blockCreate("add_src_b", source, {}) == Result::SUCCESS);

    TensorMap inputs;
    inputs["a"].requested("add_src_a", "window");
    inputs["b"].requested("add_src_b", "window");

    Blocks::Add config;
    REQUIRE(flowgraph->blockCreate("add_block", config, inputs) == Result::SUCCESS);
    REQUIRE(flowgraph->blockList().at("add_block")->state() == Block::State::Created);
    REQUIRE(flowgraph->blockList().at("add_block")->outputs().count("sum") == 1);
}

TEST_CASE_METHOD(FlowgraphFixture, "Add block handles disconnect and reconnect",
                 "[modules][add][block][lifecycle]") {
    Blocks::Window source;
    REQUIRE(flowgraph->blockCreate("add_life_a", source, {}) == Result::SUCCESS);
    REQUIRE(flowgraph->blockCreate("add_life_b", source, {}) == Result::SUCCESS);

    TensorMap inputs;
    inputs["a"].requested("add_life_a", "window");
    inputs["b"].requested("add_life_b", "window");
    REQUIRE(flowgraph->blockCreate("add_life", "add", {}, inputs) == Result::SUCCESS);

    REQUIRE(flowgraph->blockDisconnect("add_life", "b") == Result::SUCCESS);
    REQUIRE(flowgraph->blockList().at("add_life")->state() == Block::State::Incomplete);

    REQUIRE(flowgraph->blockConnect("add_life", "b", "add_life_b", "window") ==
            Result::SUCCESS);
    REQUIRE(flowgraph->blockList().at("add_life")->state() == Block::State::Created);
}
