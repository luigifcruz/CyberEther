#include <catch2/catch_test_macros.hpp>

#include <string>

#include "flowgraph_fixture.hh"

using namespace Jetstream;

TEST_CASE_METHOD(FlowgraphFixture,
                 "Decimator block reduces axis by ratio",
                 "[modules][dsp][decimator][block]") {
    Parser::Map sourceConfig;
    sourceConfig["signalDataType"] = std::string("CF32");
    sourceConfig["bufferSize"] = std::string("256");

    REQUIRE(flowgraph->blockCreate("src", "signal_generator", sourceConfig, {}) ==
            Result::SUCCESS);

    Parser::Map decimatorConfig;
    decimatorConfig["axis"] = std::string("0");
    decimatorConfig["ratio"] = std::string("4");

    TensorMap inputs;
    inputs["buffer"].requested("src", "signal");

    REQUIRE(flowgraph->blockCreate("decimator", "decimator", decimatorConfig, inputs) ==
            Result::SUCCESS);
    REQUIRE(viewBlock("decimator").state == Block::State::Created);

    const Tensor out = viewBlock("decimator").outputs.at("buffer").tensor;
    REQUIRE(out.shape(0) == 64);
}

TEST_CASE_METHOD(FlowgraphFixture,
                 "Decimator block rejects zero ratio",
                 "[modules][dsp][decimator][block][validation]") {
    Parser::Map sourceConfig;
    sourceConfig["signalDataType"] = std::string("CF32");

    REQUIRE(flowgraph->blockCreate("src", "signal_generator", sourceConfig, {}) ==
            Result::SUCCESS);

    Parser::Map decimatorConfig;
    decimatorConfig["axis"] = std::string("0");
    decimatorConfig["ratio"] = std::string("0");

    TensorMap inputs;
    inputs["buffer"].requested("src", "signal");

    REQUIRE(flowgraph->blockCreate("decimator_bad", "decimator", decimatorConfig, inputs) ==
            Result::SUCCESS);
    REQUIRE(viewBlock("decimator_bad").state == Block::State::Errored);
}
