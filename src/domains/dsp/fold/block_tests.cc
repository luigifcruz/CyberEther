#include <catch2/catch_test_macros.hpp>

#include <string>

#include "flowgraph_fixture.hh"

using namespace Jetstream;

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
    REQUIRE(flowgraph->blockList().at("fold")->state() == Block::State::Created);

    const Tensor& out = flowgraph->blockList().at("fold")->outputs().at("buffer").tensor;
    REQUIRE(out.shape(0) == 16);
}
