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
    inputs["buffer"].requested("src", "signal");

    REQUIRE(flowgraph->blockCreate("rrc", "rrc_filter", {}, inputs) == Result::SUCCESS);
    REQUIRE(viewBlock("rrc").state == Block::State::Created);

    const Tensor out = viewBlock("rrc").outputs.at("buffer").tensor;
    REQUIRE(out.dtype() == DataType::CF32);
    REQUIRE(out.shape(0) == 256);
}

TEST_CASE_METHOD(FlowgraphFixture,
                 "RRC filter block delegates config validation to its module",
                 "[modules][dsp][rrc_filter][block][validation]") {
    Parser::Map sourceConfig;
    sourceConfig["signalDataType"] = std::string("CF32");
    REQUIRE(flowgraph->blockCreate("rrc_bad_src", "signal_generator", sourceConfig, {}) ==
            Result::SUCCESS);

    TensorMap inputs;
    inputs["buffer"].requested("rrc_bad_src", "signal");

    Parser::Map config;
    config["taps"] = U64{10};
    REQUIRE(flowgraph->blockCreate("rrc_bad", "rrc_filter", config, inputs) ==
            Result::SUCCESS);
    REQUIRE(viewBlock("rrc_bad").state == Block::State::Errored);
    REQUIRE(viewBlock("rrc_bad").outputs.empty());
}
