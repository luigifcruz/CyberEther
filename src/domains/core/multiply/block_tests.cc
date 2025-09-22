#include <catch2/catch_test_macros.hpp>

#include "jetstream/domains/core/multiply/block.hh"
#include "jetstream/domains/dsp/window/block.hh"
#include "flowgraph_fixture.hh"

using namespace Jetstream;

TEST_CASE_METHOD(FlowgraphFixture, "Multiply block creates and exposes output",
                 "[modules][multiply][block]") {
    Blocks::Window source;
    source.size = 16;
    REQUIRE(flowgraph->blockCreate("mul_src_a", source, {}) == Result::SUCCESS);
    REQUIRE(flowgraph->blockCreate("mul_src_b", source, {}) == Result::SUCCESS);

    TensorMap inputs;
    inputs["a"] = {"mul_src_a", "window", {}};
    inputs["b"] = {"mul_src_b", "window", {}};

    Blocks::Multiply config;
    REQUIRE(flowgraph->blockCreate("mul_block", config, inputs) == Result::SUCCESS);
    REQUIRE(flowgraph->blockList().at("mul_block")->state() == Block::State::Created);
    REQUIRE(flowgraph->blockList().at("mul_block")->outputs().count("product") == 1);
}

TEST_CASE_METHOD(FlowgraphFixture, "Multiply block handles disconnect and reconnect",
                 "[modules][multiply][block][lifecycle]") {
    Blocks::Window source;
    REQUIRE(flowgraph->blockCreate("mul_life_a", source, {}) == Result::SUCCESS);
    REQUIRE(flowgraph->blockCreate("mul_life_b", source, {}) == Result::SUCCESS);

    TensorMap inputs;
    inputs["a"] = {"mul_life_a", "window", {}};
    inputs["b"] = {"mul_life_b", "window", {}};
    REQUIRE(flowgraph->blockCreate("mul_life", "multiply", {}, inputs) ==
            Result::SUCCESS);

    REQUIRE(flowgraph->blockDisconnect("mul_life", "a") == Result::SUCCESS);
    REQUIRE(flowgraph->blockList().at("mul_life")->state() == Block::State::Incomplete);

    REQUIRE(flowgraph->blockConnect("mul_life", "a", "mul_life_a", "window") ==
            Result::SUCCESS);
    REQUIRE(flowgraph->blockList().at("mul_life")->state() == Block::State::Created);
}
