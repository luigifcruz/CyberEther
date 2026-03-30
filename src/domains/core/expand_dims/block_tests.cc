#include <catch2/catch_test_macros.hpp>

#include "jetstream/domains/core/expand_dims/block.hh"
#include "jetstream/domains/dsp/window/block.hh"
#include "flowgraph_fixture.hh"

using namespace Jetstream;

TEST_CASE_METHOD(FlowgraphFixture, "ExpandDims block applies axis configuration",
                 "[modules][expand_dims][block]") {
    Blocks::Window source;
    source.size = 12;
    REQUIRE(flowgraph->blockCreate("expand_src", source, {}) == Result::SUCCESS);

    TensorMap inputs;
    inputs["buffer"].requested("expand_src", "window");

    Blocks::ExpandDims config;
    config.axis = 1;
    REQUIRE(flowgraph->blockCreate("expand_block", config, inputs) == Result::SUCCESS);
    REQUIRE(flowgraph->blockList().at("expand_block")->state() == Block::State::Created);
    REQUIRE(flowgraph->blockList().at("expand_block")->outputs().count("buffer") == 1);
}

TEST_CASE_METHOD(FlowgraphFixture, "ExpandDims block rejects invalid axis",
                 "[modules][expand_dims][block][validation]") {
    Blocks::Window source;
    REQUIRE(flowgraph->blockCreate("expand_bad_src", source, {}) == Result::SUCCESS);

    TensorMap inputs;
    inputs["buffer"].requested("expand_bad_src", "window");

    Blocks::ExpandDims config;
    config.axis = 10;
    REQUIRE(flowgraph->blockCreate("expand_bad", config, inputs) == Result::SUCCESS);
    REQUIRE(flowgraph->blockList().at("expand_bad")->state() == Block::State::Errored);
}
