#include <catch2/catch_test_macros.hpp>

#include <string>

#include "flowgraph_fixture.hh"

using namespace Jetstream;

TEST_CASE_METHOD(FlowgraphFixture,
                 "AM block produces F32 output",
                 "[modules][dsp][am][block]") {
    Parser::Map sourceConfig;
    sourceConfig["signalDataType"] = std::string("CF32");
    sourceConfig["bufferSize"] = std::string("256");

    REQUIRE(flowgraph->blockCreate("src", "signal_generator", sourceConfig, {}) ==
            Result::SUCCESS);

    TensorMap inputs;
    inputs["signal"].requested("src", "signal");

    REQUIRE(flowgraph->blockCreate("am", "am", {}, inputs) == Result::SUCCESS);
    REQUIRE(viewBlock("am").state == Block::State::Created);

    const Tensor out = viewBlock("am").outputs.at("signal").tensor;
    REQUIRE(out.dtype() == DataType::F32);
    REQUIRE(out.shape(0) == 256);
}

TEST_CASE_METHOD(FlowgraphFixture,
                 "AM block delegates input type validation to its module",
                 "[modules][dsp][am][block][validation]") {
    REQUIRE(flowgraph->blockCreate("am_bad_src", "signal_generator", {}, {}) ==
            Result::SUCCESS);

    TensorMap inputs;
    inputs["signal"].requested("am_bad_src", "signal");

    REQUIRE(flowgraph->blockCreate("am_bad", "am", {}, inputs) == Result::SUCCESS);
    REQUIRE(viewBlock("am_bad").state == Block::State::Errored);
    REQUIRE(viewBlock("am_bad").outputs.empty());
}
