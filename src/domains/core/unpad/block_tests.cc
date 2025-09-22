#include <catch2/catch_test_macros.hpp>

#include "jetstream/domains/core/pad/block.hh"
#include "jetstream/domains/core/unpad/block.hh"
#include "jetstream/domains/dsp/window/block.hh"
#include "flowgraph_fixture.hh"

using namespace Jetstream;

TEST_CASE_METHOD(FlowgraphFixture, "Unpad block creates and exposes both outputs",
                 "[modules][unpad][block]") {
    Blocks::Window source;
    source.size = 8;
    REQUIRE(flowgraph->blockCreate("unpad_src", source, {}) == Result::SUCCESS);

    TensorMap padInputs;
    padInputs["unpadded"] = {"unpad_src", "window", {}};
    Blocks::Pad padConfig;
    padConfig.size = 4;
    padConfig.axis = 0;
    REQUIRE(flowgraph->blockCreate("unpad_pad", padConfig, padInputs) == Result::SUCCESS);

    TensorMap inputs;
    inputs["padded"] = {"unpad_pad", "padded", {}};
    Blocks::Unpad config;
    config.size = 4;
    config.axis = 0;
    REQUIRE(flowgraph->blockCreate("unpad_block", config, inputs) == Result::SUCCESS);
    REQUIRE(flowgraph->blockList().at("unpad_block")->state() == Block::State::Created);
    REQUIRE(flowgraph->blockList().at("unpad_block")->outputs().count("unpadded") == 1);
    REQUIRE(flowgraph->blockList().at("unpad_block")->outputs().count("pad") == 1);
}

TEST_CASE_METHOD(FlowgraphFixture, "Unpad block rejects invalid axis",
                 "[modules][unpad][block][validation]") {
    Blocks::Window source;
    REQUIRE(flowgraph->blockCreate("unpad_bad_src", source, {}) == Result::SUCCESS);

    TensorMap inputs;
    inputs["padded"] = {"unpad_bad_src", "window", {}};

    Blocks::Unpad config;
    config.size = 1;
    config.axis = 5;
    REQUIRE(flowgraph->blockCreate("unpad_bad", config, inputs) == Result::SUCCESS);
    REQUIRE(flowgraph->blockList().at("unpad_bad")->state() == Block::State::Errored);
}
