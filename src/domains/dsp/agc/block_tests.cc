#include <catch2/catch_test_macros.hpp>

#include <string>

#include "flowgraph_fixture.hh"

using namespace Jetstream;

TEST_CASE_METHOD(FlowgraphFixture,
                 "AGC block wires signal input and output",
                 "[modules][dsp][agc][block]") {
    Parser::Map sourceConfig;
    sourceConfig["signalDataType"] = std::string("CF32");
    sourceConfig["bufferSize"] = std::string("128");

    REQUIRE(flowgraph->blockCreate("src", "signal_generator", sourceConfig, {}) ==
            Result::SUCCESS);

    TensorMap inputs;
    inputs["signal"].requested("src", "signal");

    REQUIRE(flowgraph->blockCreate("agc", "agc", {}, inputs) == Result::SUCCESS);
    REQUIRE(flowgraph->blockList().at("agc")->state() == Block::State::Created);

    const Tensor& out = flowgraph->blockList().at("agc")->outputs().at("signal").tensor;
    REQUIRE(out.dtype() == DataType::CF32);
    REQUIRE(out.rank() == 1);
    REQUIRE(out.shape(0) == 128);
}

TEST_CASE_METHOD(FlowgraphFixture,
                 "AGC block is incomplete without required input",
                 "[modules][dsp][agc][block][validation]") {
    auto result = flowgraph->blockCreate("agc_incomplete", "agc", {}, {});
    REQUIRE((result == Result::SUCCESS || result == Result::INCOMPLETE));
    REQUIRE(flowgraph->blockList().at("agc_incomplete")->state() ==
            Block::State::Incomplete);
}
