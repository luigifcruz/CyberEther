#include <catch2/catch_test_macros.hpp>

#include <string>

#include "flowgraph_fixture.hh"

using namespace Jetstream;

TEST_CASE_METHOD(FlowgraphFixture,
                 "FFT block creates and exposes transformed signal",
                 "[modules][dsp][fft][block]") {
    Parser::Map sourceConfig;
    sourceConfig["signalDataType"] = std::string("CF32");
    sourceConfig["bufferSize"] = std::string("64");

    REQUIRE(flowgraph->blockCreate("src", "signal_generator", sourceConfig, {}) ==
            Result::SUCCESS);

    TensorMap inputs;
    inputs["signal"].requested("src", "signal");

    REQUIRE(flowgraph->blockCreate("fft", "fft", {}, inputs) == Result::SUCCESS);
    REQUIRE(flowgraph->blockList().at("fft")->state() == Block::State::Created);

    const Tensor& out = flowgraph->blockList().at("fft")->outputs().at("signal").tensor;
    REQUIRE(out.dtype() == DataType::CF32);
    REQUIRE(out.shape(0) == 64);
}
