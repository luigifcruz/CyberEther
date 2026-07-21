#include <catch2/catch_test_macros.hpp>

#include <algorithm>

#include "jetstream/domains/core/arithmetic/block.hh"
#include "jetstream/domains/dsp/window/block.hh"
#include "flowgraph_fixture.hh"

using namespace Jetstream;

TEST_CASE_METHOD(FlowgraphFixture, "Arithmetic block creates with custom config",
                 "[modules][arithmetic][block]") {
    REQUIRE(Blocks::Arithmetic{}.axis == -1);

    Blocks::Window source;
    source.size = 8;
    REQUIRE(flowgraph->blockCreate("arith_src", source, {}) == Result::SUCCESS);

    TensorMap inputs;
    inputs["buffer"].requested("arith_src", "window");

    Blocks::Arithmetic config;
    config.operation = "add";
    config.axis = 0;
    config.squeeze = false;

    REQUIRE(flowgraph->blockCreate("arith_block", config, inputs) == Result::SUCCESS);
    const auto block = viewBlock("arith_block");
    REQUIRE(block.state == Block::State::Created);
    REQUIRE(block.outputs.count("buffer") == 1);

    const auto axis = std::find_if(block.interfaceConfigs.begin(),
                                   block.interfaceConfigs.end(),
                                   [](const auto& entry) { return entry.name == "axis"; });
    REQUIRE(axis != block.interfaceConfigs.end());
    REQUIRE(axis->format == "int:");

    Parser::Map saved;
    REQUIRE(flowgraph->blockConfig("arith_block", saved) == Result::SUCCESS);
    REQUIRE(saved.at("axis").type() == typeid(I64));
    REQUIRE(std::any_cast<I64>(saved.at("axis")) == 0);
}

TEST_CASE_METHOD(FlowgraphFixture, "Arithmetic block rejects invalid operation",
                 "[modules][arithmetic][block][validation]") {
    Blocks::Window source;
    REQUIRE(flowgraph->blockCreate("arith_bad_src", source, {}) == Result::SUCCESS);

    TensorMap inputs;
    inputs["buffer"].requested("arith_bad_src", "window");

    Blocks::Arithmetic config;
    config.operation = "invalid";
    REQUIRE(flowgraph->blockCreate("arith_bad", config, inputs) == Result::SUCCESS);
    REQUIRE(viewBlock("arith_bad").state == Block::State::Errored);
}
