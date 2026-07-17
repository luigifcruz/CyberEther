#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>
#include <string>

#include "flowgraph_fixture.hh"
#include "jetstream/domains/core/ones_tensor/block.hh"
#include "jetstream/domains/dsp/spectrum_engine/block.hh"

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
    REQUIRE(viewBlock("spec").state == Block::State::Created);

    const Tensor out = viewBlock("spec").outputs.at("buffer").tensor;
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
    REQUIRE(viewBlock("spec_bad").state == Block::State::Errored);
}

TEST_CASE_METHOD(FlowgraphFixture,
                 "Spectrum engine applies window and FFT along a non-last axis",
                 "[modules][dsp][spectrum_engine][block][axis]") {
    Blocks::OnesTensor source;
    source.shape = {5, 3};
    source.dataType = "CF32";
    REQUIRE(flowgraph->blockCreate("src", source, {}) == Result::SUCCESS);

    TensorMap inputs;
    inputs["buffer"].requested("src", "buffer");

    Blocks::SpectrumEngine config;
    config.axis = 0;
    REQUIRE(flowgraph->blockCreate("spec", config, inputs) == Result::SUCCESS);
    REQUIRE(viewBlock("spec").state == Block::State::Created);
    REQUIRE(flowgraph->compute() == Result::SUCCESS);

    const Tensor out = viewBlock("spec").outputs.at("buffer").tensor;
    REQUIRE(out.shape() == Shape{5, 3});
    REQUIRE(out.dtype() == DataType::F32);
    for (U64 row = 0; row < out.shape(0); ++row) {
        const F32 expected = out.at<F32>(row, 0);
        REQUIRE(std::isfinite(expected));
        for (U64 column = 1; column < out.shape(1); ++column) {
            REQUIRE_THAT(out.at<F32>(row, column),
                         Catch::Matchers::WithinAbs(expected, 1e-5f));
        }
    }
}
