#include <catch2/catch_test_macros.hpp>

#include <algorithm>

#include "jetstream/domains/core/expand_dims/block.hh"
#include "jetstream/domains/core/ones_tensor/block.hh"
#include "jetstream/domains/dsp/window/block.hh"
#include "flowgraph_fixture.hh"

using namespace Jetstream;

TEST_CASE_METHOD(FlowgraphFixture, "ExpandDims block applies axis configuration",
                 "[modules][expand_dims][block]") {
    REQUIRE(Blocks::ExpandDims{}.axis == -1);

    Blocks::Window source;
    source.size = 12;
    REQUIRE(flowgraph->blockCreate("expand_src", source, {}) == Result::SUCCESS);

    TensorMap inputs;
    inputs["buffer"].requested("expand_src", "window");

    Blocks::ExpandDims config;
    config.axis = 1;
    REQUIRE(flowgraph->blockCreate("expand_block", config, inputs) == Result::SUCCESS);
    const auto block = viewBlock("expand_block");
    REQUIRE(block.state == Block::State::Created);
    REQUIRE(block.outputs.count("buffer") == 1);

    const auto axis = std::find_if(block.interfaceConfigs.begin(),
                                   block.interfaceConfigs.end(),
                                   [](const auto& entry) { return entry.name == "axis"; });
    REQUIRE(axis != block.interfaceConfigs.end());
    REQUIRE(axis->format == "int:");

    Parser::Map saved;
    REQUIRE(flowgraph->blockConfig("expand_block", saved) == Result::SUCCESS);
    REQUIRE(saved.at("axis").type() == typeid(I64));
    REQUIRE(std::any_cast<I64>(saved.at("axis")) == 1);
}

TEST_CASE_METHOD(FlowgraphFixture, "ExpandDims block rejects invalid axis",
                 "[modules][expand_dims][block][validation]") {
    Blocks::Window source;
    REQUIRE(flowgraph->blockCreate("expand_bad_src", source, {}) == Result::SUCCESS);

    TensorMap inputs;
    inputs["buffer"].requested("expand_bad_src", "window");

    Blocks::ExpandDims config;
    config.axis = 10;
    REQUIRE(flowgraph->blockCreate("expand_bad", config, inputs) == Result::SUCCESS);
    REQUIRE(viewBlock("expand_bad").state == Block::State::Errored);
}

TEST_CASE_METHOD(FlowgraphFixture,
                  "ExpandDims block preserves an invalid axis for recovery",
                 "[modules][expand_dims][block][reconfigure][validation]") {
    Blocks::OnesTensor source;
    source.shape = {2, 4};
    REQUIRE(flowgraph->blockCreate("expand_recfg_src", source, {}) ==
            Result::SUCCESS);

    TensorMap inputs;
    inputs["buffer"].requested("expand_recfg_src", "buffer");

    Blocks::ExpandDims config;
    config.axis = 0;
    REQUIRE(flowgraph->blockCreate("expand_recfg", config, inputs) ==
            Result::SUCCESS);
    Parser::Map update;
    update["axis"] = I64{3};
    REQUIRE(flowgraph->blockReconfigure("expand_recfg", update) == Result::SUCCESS);

    REQUIRE(viewBlock("expand_recfg").state == Block::State::Errored);
    REQUIRE(viewBlock("expand_recfg").outputs.empty());

    Parser::Map recovery;
    recovery["axis"] = I64{0};
    REQUIRE(flowgraph->blockReconfigure("expand_recfg", recovery) == Result::SUCCESS);
    const auto recovered = viewBlock("expand_recfg");
    REQUIRE(recovered.state == Block::State::Created);
    REQUIRE(recovered.outputs.at("buffer").tensor.shape() == Shape{1, 2, 4});
}
