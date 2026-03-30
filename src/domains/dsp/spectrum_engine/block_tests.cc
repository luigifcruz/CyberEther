#include <catch2/catch_test_macros.hpp>

#include <string>

#include "flowgraph_fixture.hh"

using namespace Jetstream;

TEST_CASE_METHOD(FlowgraphFixture,
                 "Spectrum engine block creates and exposes scaled spectrum",
                 "[modules][dsp][spectrum_engine][block]") {
    Parser::Map sourceConfig;
    sourceConfig["signalDataType"] = std::string("CF32");
    sourceConfig["bufferSize"] = std::string("256");

    REQUIRE(flowgraph->blockCreate("src", "signal_generator", sourceConfig, {}) ==
            Result::SUCCESS);

    Parser::Map engineConfig;
    engineConfig["axis"] = std::string("0");
    engineConfig["enableScale"] = std::string("true");
    engineConfig["rangeMin"] = std::string("-100");
    engineConfig["rangeMax"] = std::string("0");

    TensorMap inputs;
    inputs["buffer"].requested("src", "signal");

    REQUIRE(flowgraph->blockCreate("spec", "spectrum_engine", engineConfig, inputs) ==
            Result::SUCCESS);
    REQUIRE(flowgraph->blockList().at("spec")->state() == Block::State::Created);

    const Tensor& out = flowgraph->blockList().at("spec")->outputs().at("buffer").tensor;
    REQUIRE(out.dtype() == DataType::F32);
    REQUIRE(out.shape(0) == 256);
}

TEST_CASE_METHOD(FlowgraphFixture,
                 "Spectrum engine block rejects axis out of bounds",
                 "[modules][dsp][spectrum_engine][block][validation]") {
    Parser::Map sourceConfig;
    sourceConfig["signalDataType"] = std::string("CF32");
    sourceConfig["bufferSize"] = std::string("128");

    REQUIRE(flowgraph->blockCreate("src", "signal_generator", sourceConfig, {}) ==
            Result::SUCCESS);

    Parser::Map engineConfig;
    engineConfig["axis"] = std::string("2");

    TensorMap inputs;
    inputs["buffer"].requested("src", "signal");

    REQUIRE(flowgraph->blockCreate("spec_bad", "spectrum_engine", engineConfig, inputs) ==
            Result::SUCCESS);
    REQUIRE(flowgraph->blockList().at("spec_bad")->state() == Block::State::Errored);
}
