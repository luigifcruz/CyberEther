#include <catch2/catch_test_macros.hpp>

#include "jetstream/domains/core/expand_dims/block.hh"
#include "jetstream/domains/core/squeeze_dims/block.hh"
#include "jetstream/domains/dsp/window/block.hh"
#include "flowgraph_fixture.hh"

using namespace Jetstream;

TEST_CASE_METHOD(FlowgraphFixture, "SqueezeDims block creates after expand_dims",
                 "[modules][squeeze_dims][block]") {
    Blocks::Window source;
    source.size = 8;
    REQUIRE(flowgraph->blockCreate("sq_src", source, {}) == Result::SUCCESS);

    TensorMap expandInputs;
    expandInputs["buffer"] = {"sq_src", "window", {}};
    Blocks::ExpandDims expandConfig;
    expandConfig.axis = 0;
    REQUIRE(flowgraph->blockCreate("sq_expand", expandConfig, expandInputs) ==
            Result::SUCCESS);

    TensorMap squeezeInputs;
    squeezeInputs["buffer"] = {"sq_expand", "buffer", {}};
    Blocks::SqueezeDims squeezeConfig;
    squeezeConfig.axis = 0;
    REQUIRE(flowgraph->blockCreate("sq_block", squeezeConfig, squeezeInputs) ==
            Result::SUCCESS);
    REQUIRE(flowgraph->blockList().at("sq_block")->state() == Block::State::Created);
}

TEST_CASE_METHOD(FlowgraphFixture, "SqueezeDims block rejects non-singleton axis",
                 "[modules][squeeze_dims][block][validation]") {
    Blocks::Window source;
    REQUIRE(flowgraph->blockCreate("sq_bad_src", source, {}) == Result::SUCCESS);

    TensorMap inputs;
    inputs["buffer"] = {"sq_bad_src", "window", {}};

    Blocks::SqueezeDims config;
    config.axis = 0;
    REQUIRE(flowgraph->blockCreate("sq_bad", config, inputs) == Result::SUCCESS);
    REQUIRE(flowgraph->blockList().at("sq_bad")->state() == Block::State::Errored);
}
