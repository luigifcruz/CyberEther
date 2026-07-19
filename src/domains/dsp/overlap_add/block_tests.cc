#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <string>

#include "flowgraph_fixture.hh"
#include "jetstream/domains/dsp/overlap_add/block.hh"
#include "jetstream/domains/dsp/overlap_add/module.hh"

using namespace Jetstream;

TEST_CASE("Overlap-add axis configs default to the last dimension",
          "[modules][dsp][overlap_add][config]") {
    REQUIRE(Blocks::OverlapAdd{}.axis == -1);
    REQUIRE(Modules::OverlapAdd{}.axis == -1);
}

TEST_CASE_METHOD(FlowgraphFixture,
                 "Overlap-add block creates with both required inputs",
                 "[modules][dsp][overlap_add][block]") {
    Parser::Map bufferConfig;
    bufferConfig["signalDataType"] = std::string("CF32");
    bufferConfig["bufferSize"] = std::string("128");

    Parser::Map overlapConfig;
    overlapConfig["size"] = std::string("16");

    REQUIRE(flowgraph->blockCreate("buffer_src", "signal_generator", bufferConfig, {}) ==
            Result::SUCCESS);
    REQUIRE(flowgraph->blockCreate("overlap_src", "window", overlapConfig, {}) ==
            Result::SUCCESS);

    TensorMap inputs;
    inputs["buffer"].requested("buffer_src", "signal");
    inputs["overlap"].requested("overlap_src", "window");

    REQUIRE(flowgraph->blockCreate("overlap_add", "overlap_add", {}, inputs) ==
            Result::SUCCESS);
    const auto block = viewBlock("overlap_add");
    REQUIRE(block.state == Block::State::Created);
    REQUIRE(block.outputs.contains("buffer"));

    const auto axis = std::find_if(block.interfaceConfigs.begin(),
                                   block.interfaceConfigs.end(),
                                   [](const auto& entry) { return entry.name == "axis"; });
    REQUIRE(axis != block.interfaceConfigs.end());
    REQUIRE(axis->format == "int:axis");

    Parser::Map saved;
    REQUIRE(flowgraph->blockConfig("overlap_add", saved) == Result::SUCCESS);
    REQUIRE(saved.at("axis").type() == typeid(I64));
    REQUIRE(std::any_cast<I64>(saved.at("axis")) == -1);
}
