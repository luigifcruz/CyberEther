#include <catch2/catch_test_macros.hpp>

#include <algorithm>

#include "jetstream/domains/core/invert/block.hh"
#include "jetstream/domains/dsp/window/block.hh"
#include "flowgraph_fixture.hh"

using namespace Jetstream;

TEST_CASE_METHOD(FlowgraphFixture, "Invert block creates with complex input",
                 "[modules][invert][block]") {
    Blocks::Window source;
    REQUIRE(flowgraph->blockCreate("invert_src", source, {}) == Result::SUCCESS);

    TensorMap inputs;
    inputs["signal"].requested("invert_src", "window");

    Blocks::Invert config;
    REQUIRE(flowgraph->blockCreate("invert_block", config, inputs) == Result::SUCCESS);
    REQUIRE(viewBlock("invert_block").state == Block::State::Created);
    REQUIRE(viewBlock("invert_block").outputs.count("signal") == 1);
}

TEST_CASE_METHOD(FlowgraphFixture, "Invert block input reconnect lifecycle",
                 "[modules][invert][block][lifecycle]") {
    Blocks::Window source;
    REQUIRE(flowgraph->blockCreate("invert_life_src", source, {}) == Result::SUCCESS);

    TensorMap inputs;
    inputs["signal"].requested("invert_life_src", "window");
    REQUIRE(flowgraph->blockCreate("invert_life", "invert", {}, inputs) ==
            Result::SUCCESS);

    REQUIRE(flowgraph->blockDisconnect("invert_life", "signal") == Result::SUCCESS);
    REQUIRE(viewBlock("invert_life").state == Block::State::Incomplete);

    REQUIRE(flowgraph->blockConnect("invert_life", "signal", "invert_life_src", "window")
            == Result::SUCCESS);
    REQUIRE(viewBlock("invert_life").state == Block::State::Created);
}

TEST_CASE_METHOD(FlowgraphFixture, "Integer config formats preserve signedness",
                 "[modules][invert][block][config][integer]") {
    REQUIRE(Blocks::Invert{}.axis == -1);

    Blocks::Window source;
    REQUIRE(flowgraph->blockCreate("integer_src", source, {}) == Result::SUCCESS);

    TensorMap inputs;
    inputs["signal"].requested("integer_src", "window");
    Blocks::Invert invertConfig;
    invertConfig.axis = 0;
    REQUIRE(flowgraph->blockCreate("integer_invert", invertConfig, inputs) == Result::SUCCESS);

    const auto sourceData = viewBlock("integer_src");
    const auto size = std::find_if(sourceData.interfaceConfigs.begin(),
                                   sourceData.interfaceConfigs.end(),
                                   [](const auto& entry) { return entry.name == "size"; });
    REQUIRE(size != sourceData.interfaceConfigs.end());
    REQUIRE(size->format == "uint:samples");

    const auto invertData = viewBlock("integer_invert");
    const auto axis = std::find_if(invertData.interfaceConfigs.begin(),
                                   invertData.interfaceConfigs.end(),
                                   [](const auto& entry) { return entry.name == "axis"; });
    REQUIRE(axis != invertData.interfaceConfigs.end());
    REQUIRE(axis->format == "int:");

    Parser::Map update;
    update["axis"] = I64{-1};
    REQUIRE(flowgraph->blockReconfigure("integer_invert", update) == Result::SUCCESS);

    Parser::Map saved;
    REQUIRE(flowgraph->blockConfig("integer_invert", saved) == Result::SUCCESS);
    REQUIRE(saved.at("axis").type() == typeid(I64));
    REQUIRE(std::any_cast<I64>(saved.at("axis")) == -1);
}
