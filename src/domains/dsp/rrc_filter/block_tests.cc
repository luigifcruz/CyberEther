#include <catch2/catch_test_macros.hpp>

#include <string>

#include "flowgraph_fixture.hh"

using namespace Jetstream;

TEST_CASE_METHOD(FlowgraphFixture,
                 "RRC filter block creates with valid defaults",
                 "[modules][dsp][rrc_filter][block]") {
    Parser::Map sourceConfig;
    sourceConfig["signalDataType"] = std::string("CF32");
    sourceConfig["bufferSize"] = std::string("256");

    REQUIRE(flowgraph->blockCreate("src", "signal_generator", sourceConfig, {}) ==
            Result::SUCCESS);

    TensorMap inputs;
    inputs["buffer"] = {"src", "signal", {}};

    REQUIRE(flowgraph->blockCreate("rrc", "rrc_filter", {}, inputs) == Result::SUCCESS);
    REQUIRE(flowgraph->blockList().at("rrc")->state() == Block::State::Created);

    const Tensor& out = flowgraph->blockList().at("rrc")->outputs().at("buffer").tensor;
    REQUIRE(out.dtype() == DataType::CF32);
    REQUIRE(out.shape(0) == 256);
}
