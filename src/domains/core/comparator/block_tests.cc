#include <catch2/catch_test_macros.hpp>

#include "jetstream/domains/core/comparator/block.hh"
#include "jetstream/domains/dsp/window/block.hh"
#include "flowgraph_fixture.hh"

using namespace Jetstream;

TEST_CASE_METHOD(FlowgraphFixture, "Comparator block creates and exposes output",
                 "[modules][comparator][block]") {
    Blocks::Window source;
    source.size = 16;
    REQUIRE(flowgraph->blockCreate("cmp_src_a", source, {}) == Result::SUCCESS);
    REQUIRE(flowgraph->blockCreate("cmp_src_b", source, {}) == Result::SUCCESS);

    TensorMap inputs;
    inputs["input0"].requested("cmp_src_a", "window");
    inputs["input1"].requested("cmp_src_b", "window");

    Blocks::Comparator config;
    config.inputCount = 2;
    REQUIRE(flowgraph->blockCreate("cmp_block", config, inputs) == Result::SUCCESS);
    REQUIRE(viewBlock("cmp_block").state == Block::State::Created);
    REQUIRE(viewBlock("cmp_block").outputs.count("error") == 1);
}

TEST_CASE_METHOD(FlowgraphFixture, "Comparator block handles disconnect and reconnect",
                 "[modules][comparator][block][lifecycle]") {
    Blocks::Window source;
    REQUIRE(flowgraph->blockCreate("cmp_life_a", source, {}) == Result::SUCCESS);
    REQUIRE(flowgraph->blockCreate("cmp_life_b", source, {}) == Result::SUCCESS);

    TensorMap inputs;
    inputs["input0"].requested("cmp_life_a", "window");
    inputs["input1"].requested("cmp_life_b", "window");

    Blocks::Comparator config;
    config.inputCount = 2;
    REQUIRE(flowgraph->blockCreate("cmp_life", config, inputs) == Result::SUCCESS);

    REQUIRE(flowgraph->blockDisconnect("cmp_life", "input1") == Result::SUCCESS);
    REQUIRE(viewBlock("cmp_life").state == Block::State::Incomplete);

    REQUIRE(flowgraph->blockConnect("cmp_life", "input1", "cmp_life_b", "window") ==
            Result::SUCCESS);
    REQUIRE(viewBlock("cmp_life").state == Block::State::Created);
}

TEST_CASE_METHOD(FlowgraphFixture, "Comparator block accepts F32 tolerance reconfigure",
                 "[modules][comparator][block][reconfigure]") {
    Blocks::Window source;
    source.size = 16;
    REQUIRE(flowgraph->blockCreate("cmp_cfg_a", source, {}) == Result::SUCCESS);
    REQUIRE(flowgraph->blockCreate("cmp_cfg_b", source, {}) == Result::SUCCESS);

    TensorMap inputs;
    inputs["input0"].requested("cmp_cfg_a", "window");
    inputs["input1"].requested("cmp_cfg_b", "window");

    Blocks::Comparator config;
    REQUIRE(flowgraph->blockCreate("cmp_cfg", config, inputs) == Result::SUCCESS);

    Parser::Map update = viewBlock("cmp_cfg").config;
    update["tolerance"] = F32{0.25f};

    REQUIRE(flowgraph->blockReconfigure("cmp_cfg", update) == Result::SUCCESS);
    REQUIRE(viewBlock("cmp_cfg").state == Block::State::Created);
}
