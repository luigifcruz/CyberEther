#include <catch2/catch_test_macros.hpp>

#include <string>

#include "flowgraph_fixture.hh"

using namespace Jetstream;

TEST_CASE_METHOD(FlowgraphFixture,
                 "Amplitude block converts CF32 signal to F32",
                 "[modules][dsp][amplitude][block]") {
    Parser::Map sourceConfig;
    sourceConfig["signalDataType"] = std::string("CF32");
    sourceConfig["bufferSize"] = std::string("128");

    REQUIRE(flowgraph->blockCreate("src", "signal_generator", sourceConfig, {}) ==
            Result::SUCCESS);

    TensorMap inputs;
    inputs["signal"].requested("src", "signal");

    REQUIRE(flowgraph->blockCreate("amp", "amplitude", {}, inputs) == Result::SUCCESS);
    REQUIRE(flowgraph->blockList().at("amp")->state() == Block::State::Created);

    const Tensor& out = flowgraph->blockList().at("amp")->outputs().at("signal").tensor;
    REQUIRE(out.dtype() == DataType::F32);
    REQUIRE(out.shape(0) == 128);
}
