#include <catch2/catch_test_macros.hpp>

#include <string>

#include "flowgraph_fixture.hh"

using namespace Jetstream;

TEST_CASE_METHOD(FlowgraphFixture,
                 "PSK demod block creates and exposes output",
                 "[modules][dsp][psk_demod][block]") {
    Parser::Map sourceConfig;
    sourceConfig["signalDataType"] = std::string("CF32");
    sourceConfig["bufferSize"] = std::string("1024");

    REQUIRE(flowgraph->blockCreate("src", "signal_generator", sourceConfig, {}) ==
            Result::SUCCESS);

    TensorMap inputs;
    inputs["signal"] = {"src", "signal", {}};

    REQUIRE(flowgraph->blockCreate("demod", "psk_demod", {}, inputs) == Result::SUCCESS);
    REQUIRE(flowgraph->blockList().at("demod")->state() == Block::State::Created);

    const Tensor& out = flowgraph->blockList().at("demod")->outputs().at("signal").tensor;
    REQUIRE(out.rank() == 1);
    REQUIRE(out.dtype() == DataType::CF32);
}
