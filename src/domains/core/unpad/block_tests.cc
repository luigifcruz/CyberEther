#include <catch2/catch_test_macros.hpp>

#include <algorithm>

#include "jetstream/domains/core/pad/block.hh"
#include "jetstream/domains/core/unpad/block.hh"
#include "jetstream/domains/dsp/window/block.hh"
#include "flowgraph_fixture.hh"

using namespace Jetstream;

TEST_CASE_METHOD(FlowgraphFixture, "Unpad block creates and exposes both outputs",
                 "[modules][unpad][block]") {
    REQUIRE(Blocks::Unpad{}.axis == -1);

    Blocks::Window source;
    source.size = 8;
    REQUIRE(flowgraph->blockCreate("unpad_src", source, {}) == Result::SUCCESS);

    TensorMap padInputs;
    padInputs["unpadded"].requested("unpad_src", "window");
    Blocks::Pad padConfig;
    padConfig.size = 4;
    padConfig.axis = 0;
    REQUIRE(flowgraph->blockCreate("unpad_pad", padConfig, padInputs) == Result::SUCCESS);

    TensorMap inputs;
    inputs["padded"].requested("unpad_pad", "padded");
    Blocks::Unpad config;
    config.size = 4;
    config.axis = 0;
    REQUIRE(flowgraph->blockCreate("unpad_block", config, inputs) == Result::SUCCESS);
    const auto block = viewBlock("unpad_block");
    REQUIRE(block.state == Block::State::Created);
    REQUIRE(block.outputs.count("unpadded") == 1);
    REQUIRE(block.outputs.count("pad") == 1);

    const auto axis = std::find_if(block.interfaceConfigs.begin(),
                                   block.interfaceConfigs.end(),
                                   [](const auto& entry) { return entry.name == "axis"; });
    REQUIRE(axis != block.interfaceConfigs.end());
    REQUIRE(axis->format == "int:");

    Parser::Map saved;
    REQUIRE(flowgraph->blockConfig("unpad_block", saved) == Result::SUCCESS);
    REQUIRE(saved.at("axis").type() == typeid(I64));
    REQUIRE(std::any_cast<I64>(saved.at("axis")) == 0);
}

TEST_CASE_METHOD(FlowgraphFixture, "Unpad block rejects invalid axis",
                 "[modules][unpad][block][validation]") {
    Blocks::Window source;
    REQUIRE(flowgraph->blockCreate("unpad_bad_src", source, {}) == Result::SUCCESS);

    TensorMap inputs;
    inputs["padded"].requested("unpad_bad_src", "window");

    Blocks::Unpad config;
    config.size = 1;
    config.axis = 5;
    REQUIRE(flowgraph->blockCreate("unpad_bad", config, inputs) == Result::SUCCESS);
    REQUIRE(viewBlock("unpad_bad").state == Block::State::Errored);
}
