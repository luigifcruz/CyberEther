#include <catch2/catch_test_macros.hpp>

#include <algorithm>

#include "jetstream/domains/core/expand_dims/block.hh"
#include "jetstream/domains/core/squeeze_dims/block.hh"
#include "jetstream/domains/dsp/window/block.hh"
#include "flowgraph_fixture.hh"

using namespace Jetstream;

TEST_CASE_METHOD(FlowgraphFixture, "SqueezeDims block creates after expand_dims",
                 "[modules][squeeze_dims][block]") {
    REQUIRE(Blocks::SqueezeDims{}.axis == -1);

    Blocks::Window source;
    source.size = 8;
    REQUIRE(flowgraph->blockCreate("sq_src", source, {}) == Result::SUCCESS);

    TensorMap expandInputs;
    expandInputs["buffer"].requested("sq_src", "window");
    Blocks::ExpandDims expandConfig;
    expandConfig.axis = 0;
    REQUIRE(flowgraph->blockCreate("sq_expand", expandConfig, expandInputs) ==
            Result::SUCCESS);

    TensorMap squeezeInputs;
    squeezeInputs["buffer"].requested("sq_expand", "buffer");
    Blocks::SqueezeDims squeezeConfig;
    squeezeConfig.axis = 0;
    REQUIRE(flowgraph->blockCreate("sq_block", squeezeConfig, squeezeInputs) ==
            Result::SUCCESS);
    const auto block = viewBlock("sq_block");
    REQUIRE(block.state == Block::State::Created);

    const auto axis = std::find_if(block.interfaceConfigs.begin(),
                                   block.interfaceConfigs.end(),
                                   [](const auto& entry) { return entry.name == "axis"; });
    REQUIRE(axis != block.interfaceConfigs.end());
    REQUIRE(axis->format == "int:");

    Parser::Map saved;
    REQUIRE(flowgraph->blockConfig("sq_block", saved) == Result::SUCCESS);
    REQUIRE(saved.at("axis").type() == typeid(I64));
    REQUIRE(std::any_cast<I64>(saved.at("axis")) == 0);
}

TEST_CASE_METHOD(FlowgraphFixture, "SqueezeDims block rejects non-singleton axis",
                 "[modules][squeeze_dims][block][validation]") {
    Blocks::Window source;
    REQUIRE(flowgraph->blockCreate("sq_bad_src", source, {}) == Result::SUCCESS);

    TensorMap inputs;
    inputs["buffer"].requested("sq_bad_src", "window");

    Blocks::SqueezeDims config;
    config.axis = 0;
    REQUIRE(flowgraph->blockCreate("sq_bad", config, inputs) == Result::SUCCESS);
    REQUIRE(viewBlock("sq_bad").state == Block::State::Errored);
}
