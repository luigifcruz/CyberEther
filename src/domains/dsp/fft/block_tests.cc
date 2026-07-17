#include <catch2/catch_test_macros.hpp>

#include <string>

#include "flowgraph_fixture.hh"
#include "jetstream/domains/dsp/fft/block.hh"

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
    REQUIRE(viewBlock("fft").state == Block::State::Created);

    const Tensor out = viewBlock("fft").outputs.at("signal").tensor;
    REQUIRE(out.dtype() == DataType::CF32);
    REQUIRE(out.shape(0) == 64);
}

TEST_CASE_METHOD(FlowgraphFixture,
                 "FFT block partial recreation preserves unrelated settings",
                 "[modules][dsp][fft][block][reconfigure]") {
    Parser::Map sourceConfig;
    sourceConfig["signalDataType"] = std::string("CF32");
    sourceConfig["bufferSize"] = std::string("64");
    REQUIRE(flowgraph->blockCreate("src", "signal_generator", sourceConfig, {}) ==
            Result::SUCCESS);

    TensorMap inputs;
    inputs["signal"].requested("src", "signal");

    Blocks::Fft config;
    config.forward = false;
    config.axis = 0;
    config.invert = true;
    REQUIRE(flowgraph->blockCreate("fft", config, inputs) == Result::SUCCESS);

    Parser::Map update;
    update["axis"] = I64{-1};
    REQUIRE(flowgraph->blockReconfigure("fft", update) == Result::SUCCESS);
    REQUIRE(viewBlock("fft").state == Block::State::Created);

    Parser::Map savedMap;
    REQUIRE(flowgraph->blockConfig("fft", savedMap) == Result::SUCCESS);
    Blocks::Fft saved;
    REQUIRE(saved.deserialize(savedMap) == Result::SUCCESS);
    REQUIRE_FALSE(saved.forward);
    REQUIRE(saved.axis == -1);
    REQUIRE(saved.invert);

    Parser::Map invalidUpdate;
    invalidUpdate["axis"] = I64{1};
    REQUIRE(flowgraph->blockReconfigure("fft", invalidUpdate) == Result::ERROR);
    REQUIRE(viewBlock("fft").state == Block::State::Created);

    Parser::Map unchangedMap;
    REQUIRE(flowgraph->blockConfig("fft", unchangedMap) == Result::SUCCESS);
    Blocks::Fft unchanged;
    REQUIRE(unchanged.deserialize(unchangedMap) == Result::SUCCESS);
    REQUIRE(unchanged.axis == -1);
}
