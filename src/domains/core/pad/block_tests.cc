#include <catch2/catch_test_macros.hpp>

#include <algorithm>

#include "jetstream/domains/core/pad/block.hh"
#include "jetstream/domains/dsp/window/block.hh"
#include "flowgraph_fixture.hh"

using namespace Jetstream;

TEST_CASE_METHOD(FlowgraphFixture, "Pad block creates with axis and size",
                 "[modules][pad][block]") {
    REQUIRE(Blocks::Pad{}.axis == -1);

    Blocks::Window source;
    source.size = 8;
    REQUIRE(flowgraph->blockCreate("pad_src", source, {}) == Result::SUCCESS);

    TensorMap inputs;
    inputs["unpadded"].requested("pad_src", "window");

    Blocks::Pad config;
    config.size = 4;
    config.axis = 0;
    REQUIRE(flowgraph->blockCreate("pad_block", config, inputs) == Result::SUCCESS);
    const auto block = viewBlock("pad_block");
    REQUIRE(block.state == Block::State::Created);
    REQUIRE(block.outputs.count("padded") == 1);

    const auto axis = std::find_if(block.interfaceConfigs.begin(),
                                   block.interfaceConfigs.end(),
                                   [](const auto& entry) { return entry.name == "axis"; });
    REQUIRE(axis != block.interfaceConfigs.end());
    REQUIRE(axis->format == "int:");

    Parser::Map saved;
    REQUIRE(flowgraph->blockConfig("pad_block", saved) == Result::SUCCESS);
    REQUIRE(saved.at("axis").type() == typeid(I64));
    REQUIRE(std::any_cast<I64>(saved.at("axis")) == 0);
}

TEST_CASE_METHOD(FlowgraphFixture, "Pad block rejects invalid axis",
                 "[modules][pad][block][validation]") {
    Blocks::Window source;
    REQUIRE(flowgraph->blockCreate("pad_bad_src", source, {}) == Result::SUCCESS);

    TensorMap inputs;
    inputs["unpadded"].requested("pad_bad_src", "window");

    Blocks::Pad config;
    config.axis = 5;
    config.size = 1;
    REQUIRE(flowgraph->blockCreate("pad_bad", config, inputs) == Result::SUCCESS);
    REQUIRE(viewBlock("pad_bad").state == Block::State::Errored);
}
