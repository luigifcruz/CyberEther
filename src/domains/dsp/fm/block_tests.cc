#include <catch2/catch_test_macros.hpp>

#include <any>
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
    inputs["signal"].requested("src", "signal");

    REQUIRE(flowgraph->blockCreate("fm", "fm", {}, inputs) == Result::SUCCESS);
    REQUIRE(viewBlock("fm").state == Block::State::Created);

    const Tensor out = viewBlock("fm").outputs.at("signal").tensor;
    REQUIRE(out.dtype() == DataType::F32);
    REQUIRE(out.shape(0) == 128);
}

TEST_CASE_METHOD(FlowgraphFixture,
                 "FM block delegates input type validation to its module",
                 "[modules][dsp][fm][block][validation]") {
    REQUIRE(flowgraph->blockCreate("fm_bad_src", "signal_generator", {}, {}) ==
            Result::SUCCESS);

    TensorMap inputs;
    inputs["signal"].requested("fm_bad_src", "signal");

    REQUIRE(flowgraph->blockCreate("fm_bad", "fm", {}, inputs) == Result::SUCCESS);
    REQUIRE(viewBlock("fm_bad").state == Block::State::Errored);
    REQUIRE(viewBlock("fm_bad").outputs.empty());
}

TEST_CASE_METHOD(FlowgraphFixture,
                 "FM block exposes wideband stereo output",
                 "[modules][dsp][fm][block][stereo]") {
    Parser::Map sourceConfig;
    sourceConfig["signalDataType"] = std::string("CF32");
    sourceConfig["bufferSize"] = std::string("128");
    REQUIRE(flowgraph->blockCreate("src", "signal_generator", sourceConfig, {}) ==
            Result::SUCCESS);

    TensorMap inputs;
    inputs["signal"].requested("src", "signal");
    Parser::Map config;
    config["mode"] = std::string("wide");
    config["deemphasis"] = std::string("50us");
    REQUIRE(flowgraph->blockCreate("fm_wide", "fm", config, inputs) ==
            Result::SUCCESS);
    REQUIRE(viewBlock("fm_wide").state == Block::State::Created);

    const Tensor output = viewBlock("fm_wide").outputs.at("signal").tensor;
    REQUIRE(output.shape() == Shape{128, 2});
    REQUIRE(std::any_cast<Index>(output.attribute("sampleAxis")) == Index{0});
    REQUIRE(std::any_cast<Index>(output.attribute("channelAxis")) == Index{1});
}
