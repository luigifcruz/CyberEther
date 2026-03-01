#include <catch2/catch_test_macros.hpp>

#include "jetstream/domains/core/reshape/block.hh"
#include "jetstream/domains/dsp/window/block.hh"
#include "flowgraph_fixture.hh"

using namespace Jetstream;

TEST_CASE_METHOD(FlowgraphFixture, "Reshape block creates with target shape",
                 "[modules][reshape][block]") {
    Blocks::Window source;
    source.size = 8;
    REQUIRE(flowgraph->blockCreate("reshape_src", source, {}) == Result::SUCCESS);

    TensorMap inputs;
    inputs["buffer"] = {"reshape_src", "window", {}};

    Blocks::Reshape config;
    config.shape = "[2, 4]";
    config.contiguous = true;
    REQUIRE(flowgraph->blockCreate("reshape_block", config, inputs) == Result::SUCCESS);
    REQUIRE(flowgraph->blockList().at("reshape_block")->state() == Block::State::Created);
    REQUIRE(flowgraph->blockList().at("reshape_block")->outputs().count("buffer") == 1);
}

TEST_CASE_METHOD(FlowgraphFixture, "Reshape block rejects invalid target shape",
                 "[modules][reshape][block][validation]") {
    Blocks::Window source;
    source.size = 8;
    REQUIRE(flowgraph->blockCreate("reshape_bad_src", source, {}) == Result::SUCCESS);

    TensorMap inputs;
    inputs["buffer"] = {"reshape_bad_src", "window", {}};

    Blocks::Reshape config;
    config.shape = "[7]";
    REQUIRE(flowgraph->blockCreate("reshape_bad", config, inputs) == Result::SUCCESS);
    REQUIRE(flowgraph->blockList().at("reshape_bad")->state() == Block::State::Errored);
}
