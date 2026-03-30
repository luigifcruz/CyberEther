#include <catch2/catch_test_macros.hpp>

#include "jetstream/domains/core/cast/block.hh"
#include "jetstream/domains/dsp/window/block.hh"
#include "flowgraph_fixture.hh"

using namespace Jetstream;

TEST_CASE_METHOD(FlowgraphFixture, "Cast block reports error for unsupported source type",
                 "[modules][cast][block]") {
    Blocks::Window source;
    REQUIRE(flowgraph->blockCreate("cast_src", source, {}) == Result::SUCCESS);

    TensorMap inputs;
    inputs["buffer"].requested("cast_src", "window");

    Blocks::Cast config;
    config.outputType = "CF32";
    REQUIRE(flowgraph->blockCreate("cast_block", config, inputs) == Result::SUCCESS);
    REQUIRE(flowgraph->blockList().at("cast_block")->state() == Block::State::Errored);
}

TEST_CASE_METHOD(FlowgraphFixture, "Cast block rejects invalid output type",
                 "[modules][cast][block][validation]") {
    Blocks::Window source;
    REQUIRE(flowgraph->blockCreate("cast_bad_src", source, {}) == Result::SUCCESS);

    TensorMap inputs;
    inputs["buffer"].requested("cast_bad_src", "window");

    Blocks::Cast config;
    config.outputType = "INVALID";
    REQUIRE(flowgraph->blockCreate("cast_bad", config, inputs) == Result::SUCCESS);
    REQUIRE(flowgraph->blockList().at("cast_bad")->state() == Block::State::Errored);
}
