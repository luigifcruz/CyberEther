#include <catch2/catch_test_macros.hpp>

#include "flowgraph_fixture.hh"
#include "jetstream/domains/dsp/squelch/block.hh"

using namespace Jetstream;

TEST_CASE_METHOD(FlowgraphFixture,
                 "Squelch block delegates threshold validation to its module",
                 "[modules][dsp][squelch][block][validation]") {
    REQUIRE(flowgraph->blockCreate("squelch_src", "signal_generator", {}, {}) ==
            Result::SUCCESS);

    TensorMap inputs;
    inputs["signal"].requested("squelch_src", "signal");

    Blocks::Squelch config;
    config.threshold = -0.1f;
    REQUIRE(flowgraph->blockCreate("squelch_bad", config, inputs) == Result::SUCCESS);
    REQUIRE(viewBlock("squelch_bad").state == Block::State::Errored);
    REQUIRE(viewBlock("squelch_bad").outputs.empty());
}
