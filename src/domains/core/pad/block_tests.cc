#include <catch2/catch_test_macros.hpp>

#include "jetstream/domains/core/pad/block.hh"
#include "jetstream/domains/dsp/window/block.hh"
#include "flowgraph_fixture.hh"

using namespace Jetstream;

TEST_CASE_METHOD(FlowgraphFixture, "Pad block creates with axis and size",
                 "[modules][pad][block]") {
    Blocks::Window source;
    source.size = 8;
    REQUIRE(flowgraph->blockCreate("pad_src", source, {}) == Result::SUCCESS);

    TensorMap inputs;
    inputs["unpadded"].requested("pad_src", "window");

    Blocks::Pad config;
    config.size = 4;
    config.axis = 0;
    REQUIRE(flowgraph->blockCreate("pad_block", config, inputs) == Result::SUCCESS);
    REQUIRE(flowgraph->blockList().at("pad_block")->state() == Block::State::Created);
    REQUIRE(flowgraph->blockList().at("pad_block")->outputs().count("padded") == 1);
}

TEST_CASE_METHOD(FlowgraphFixture, "Pad block rejects invalid axis",
                 "[modules][pad][block][validation]") {
    Blocks::Window source;
    REQUIRE(flowgraph->blockCreate("pad_bad_src", source, {}) == Result::SUCCESS);

    TensorMap inputs;
    inputs["unpadded"].requested("pad_bad_src", "window");

    Blocks::Pad config;
    config.axis = 5;
    config.size = 1;
    REQUIRE(flowgraph->blockCreate("pad_bad", config, inputs) == Result::SUCCESS);
    REQUIRE(flowgraph->blockList().at("pad_bad")->state() == Block::State::Errored);
}
