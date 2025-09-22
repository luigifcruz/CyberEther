#include <catch2/catch_test_macros.hpp>

#include "jetstream/domains/core/multiply_constant/block.hh"
#include "jetstream/domains/dsp/window/block.hh"
#include "flowgraph_fixture.hh"

using namespace Jetstream;

TEST_CASE_METHOD(FlowgraphFixture, "MultiplyConstant block creates with custom constant",
                 "[modules][multiply_constant][block]") {
    Blocks::Window source;
    REQUIRE(flowgraph->blockCreate("mulc_src", source, {}) == Result::SUCCESS);

    TensorMap inputs;
    inputs["factor"] = {"mulc_src", "window", {}};

    Blocks::MultiplyConstant config;
    config.constant = 0.25f;
    REQUIRE(flowgraph->blockCreate("mulc_block", config, inputs) == Result::SUCCESS);
    REQUIRE(flowgraph->blockList().at("mulc_block")->state() == Block::State::Created);
    REQUIRE(flowgraph->blockList().at("mulc_block")->outputs().count("product") == 1);
}

TEST_CASE_METHOD(FlowgraphFixture, "MultiplyConstant block reconnects input",
                 "[modules][multiply_constant][block][lifecycle]") {
    Blocks::Window source;
    REQUIRE(flowgraph->blockCreate("mulc_life_src", source, {}) == Result::SUCCESS);

    TensorMap inputs;
    inputs["factor"] = {"mulc_life_src", "window", {}};
    REQUIRE(flowgraph->blockCreate("mulc_life", "multiply_constant", {}, inputs) ==
            Result::SUCCESS);

    REQUIRE(flowgraph->blockDisconnect("mulc_life", "factor") == Result::SUCCESS);
    REQUIRE(flowgraph->blockList().at("mulc_life")->state() == Block::State::Incomplete);

    REQUIRE(flowgraph->blockConnect("mulc_life", "factor", "mulc_life_src", "window") ==
            Result::SUCCESS);
    REQUIRE(flowgraph->blockList().at("mulc_life")->state() == Block::State::Created);
}
