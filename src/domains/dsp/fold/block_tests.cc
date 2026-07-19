#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <string>

#include "flowgraph_fixture.hh"
#include "jetstream/domains/dsp/fold/block.hh"
#include "jetstream/domains/dsp/fold/module.hh"

using namespace Jetstream;

TEST_CASE("Fold axis configs default to the last dimension",
          "[modules][dsp][fold][config]") {
    REQUIRE(Blocks::Fold{}.axis == -1);
    REQUIRE(Modules::Fold{}.axis == -1);
}

TEST_CASE_METHOD(FlowgraphFixture,
                 "Fold block creates for valid axis and size",
                 "[modules][dsp][fold][block]") {
    Parser::Map sourceConfig;
    sourceConfig["signalDataType"] = std::string("F32");
    sourceConfig["bufferSize"] = std::string("64");

    REQUIRE(flowgraph->blockCreate("src", "signal_generator", sourceConfig, {}) ==
            Result::SUCCESS);

    Parser::Map foldConfig;
    foldConfig["axis"] = std::string("0");
    foldConfig["offset"] = std::string("0");
    foldConfig["size"] = std::string("16");

    TensorMap inputs;
    inputs["buffer"].requested("src", "signal");

    REQUIRE(flowgraph->blockCreate("fold", "fold", foldConfig, inputs) == Result::SUCCESS);
    const auto block = viewBlock("fold");
    REQUIRE(block.state == Block::State::Created);

    const auto axis = std::find_if(block.interfaceConfigs.begin(),
                                   block.interfaceConfigs.end(),
                                   [](const auto& entry) { return entry.name == "axis"; });
    REQUIRE(axis != block.interfaceConfigs.end());
    REQUIRE(axis->format == "int:axis");

    Parser::Map saved;
    REQUIRE(flowgraph->blockConfig("fold", saved) == Result::SUCCESS);
    REQUIRE(saved.at("axis").type() == typeid(I64));
    REQUIRE(std::any_cast<I64>(saved.at("axis")) == 0);

    const Tensor out = block.outputs.at("buffer").tensor;
    REQUIRE(out.shape(0) == 16);
}
