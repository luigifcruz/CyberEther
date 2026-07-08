#include <catch2/catch_test_macros.hpp>

#include "flowgraph_fixture.hh"
#include "jetstream/domains/core/flatten/block.hh"
#include "jetstream/domains/dsp/window/block.hh"

using namespace Jetstream;

TEST_CASE_METHOD(FlowgraphFixture, "Flatten block creates",
                 "[modules][flatten][block]") {
    Blocks::Window source;
    source.size = 8;
    REQUIRE(flowgraph->blockCreate("flatten_src", source, {}) == Result::SUCCESS);

    TensorMap inputs;
    inputs["buffer"].requested("flatten_src", "window");

    Blocks::Flatten config;
    REQUIRE(flowgraph->blockCreate("flatten_block", config, inputs) == Result::SUCCESS);
    REQUIRE(viewBlock("flatten_block").state == Block::State::Created);
    REQUIRE(viewBlock("flatten_block").outputs.count("buffer") == 1);
}
