#include <catch2/catch_test_macros.hpp>

#include "jetstream/domains/core/arithmetic/block.hh"
#include "jetstream/domains/dsp/window/block.hh"
#include "flowgraph_fixture.hh"

using namespace Jetstream;

TEST_CASE_METHOD(FlowgraphFixture, "Arithmetic block creates with custom config",
                 "[modules][arithmetic][block]") {
    Blocks::Window source;
    source.size = 8;
    REQUIRE(flowgraph->blockCreate("arith_src", source, {}) == Result::SUCCESS);

    TensorMap inputs;
    inputs["buffer"] = {"arith_src", "window", {}};

    Blocks::Arithmetic config;
    config.operation = "add";
    config.axis = 0;
    config.squeeze = false;

    REQUIRE(flowgraph->blockCreate("arith_block", config, inputs) == Result::SUCCESS);
    REQUIRE(flowgraph->blockList().at("arith_block")->state() == Block::State::Created);
    REQUIRE(flowgraph->blockList().at("arith_block")->outputs().count("buffer") == 1);
}

TEST_CASE_METHOD(FlowgraphFixture, "Arithmetic block rejects invalid operation",
                 "[modules][arithmetic][block][validation]") {
    Blocks::Window source;
    REQUIRE(flowgraph->blockCreate("arith_bad_src", source, {}) == Result::SUCCESS);

    TensorMap inputs;
    inputs["buffer"] = {"arith_bad_src", "window", {}};

    Blocks::Arithmetic config;
    config.operation = "invalid";
    REQUIRE(flowgraph->blockCreate("arith_bad", config, inputs) == Result::SUCCESS);
    REQUIRE(flowgraph->blockList().at("arith_bad")->state() == Block::State::Errored);
}
