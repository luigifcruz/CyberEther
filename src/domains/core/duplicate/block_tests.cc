#include <catch2/catch_test_macros.hpp>

#include "jetstream/domains/core/duplicate/block.hh"
#include "jetstream/domains/dsp/window/block.hh"
#include "flowgraph_fixture.hh"

using namespace Jetstream;

TEST_CASE_METHOD(FlowgraphFixture, "Duplicate block creates and exposes buffer",
                 "[modules][duplicate][block]") {
    Blocks::Window source;
    REQUIRE(flowgraph->blockCreate("dup_src", source, {}) == Result::SUCCESS);

    TensorMap inputs;
    inputs["buffer"].requested("dup_src", "window");

    Blocks::Duplicate config;
    config.hostAccessible = true;
    REQUIRE(flowgraph->blockCreate("dup_block", config, inputs) == Result::SUCCESS);
    REQUIRE(flowgraph->blockList().at("dup_block")->state() == Block::State::Created);
    REQUIRE(flowgraph->blockList().at("dup_block")->outputs().count("buffer") == 1);
}

TEST_CASE_METHOD(FlowgraphFixture, "Duplicate block reconnects input",
                 "[modules][duplicate][block][lifecycle]") {
    Blocks::Window source;
    REQUIRE(flowgraph->blockCreate("dup_life_src", source, {}) == Result::SUCCESS);

    TensorMap inputs;
    inputs["buffer"].requested("dup_life_src", "window");
    REQUIRE(flowgraph->blockCreate("dup_life", "duplicate", {}, inputs) ==
            Result::SUCCESS);

    REQUIRE(flowgraph->blockDisconnect("dup_life", "buffer") == Result::SUCCESS);
    REQUIRE(flowgraph->blockList().at("dup_life")->state() == Block::State::Incomplete);

    REQUIRE(flowgraph->blockConnect("dup_life", "buffer", "dup_life_src", "window") ==
            Result::SUCCESS);
    REQUIRE(flowgraph->blockList().at("dup_life")->state() == Block::State::Created);
}
