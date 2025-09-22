#include <catch2/catch_test_macros.hpp>

#include <string>

#include "flowgraph_fixture.hh"

using namespace Jetstream;

TEST_CASE_METHOD(FlowgraphFixture,
                 "FM block creates and outputs demodulated stream",
                 "[modules][dsp][fm][block]") {
    Parser::Map sourceConfig;
    sourceConfig["signalDataType"] = std::string("CF32");
    sourceConfig["bufferSize"] = std::string("128");

    REQUIRE(flowgraph->blockCreate("src", "signal_generator", sourceConfig, {}) ==
            Result::SUCCESS);

    TensorMap inputs;
    inputs["signal"] = {"src", "signal", {}};

    REQUIRE(flowgraph->blockCreate("fm", "fm", {}, inputs) == Result::SUCCESS);
    REQUIRE(flowgraph->blockList().at("fm")->state() == Block::State::Created);

    const Tensor& out = flowgraph->blockList().at("fm")->outputs().at("signal").tensor;
    REQUIRE(out.dtype() == DataType::F32);
    REQUIRE(out.shape(0) == 128);
}
