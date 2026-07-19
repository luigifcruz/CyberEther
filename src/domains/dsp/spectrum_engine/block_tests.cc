#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <algorithm>
#include <cmath>
#include <string>

#include "flowgraph_fixture.hh"
#include "jetstream/domains/core/ones_tensor/block.hh"
#include "jetstream/domains/dsp/spectrum_engine/block.hh"

using namespace Jetstream;

TEST_CASE("Spectrum engine axis defaults to the last dimension",
          "[modules][dsp][spectrum_engine][config]") {
    REQUIRE(Blocks::SpectrumEngine{}.axis == -1);
}

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
    const auto block = viewBlock("spec");
    REQUIRE(block.state == Block::State::Created);

    const auto axis = std::find_if(block.interfaceConfigs.begin(),
                                   block.interfaceConfigs.end(),
                                   [](const auto& entry) { return entry.name == "axis"; });
    REQUIRE(axis != block.interfaceConfigs.end());
    REQUIRE(axis->format == "int:");

    Parser::Map saved;
    REQUIRE(flowgraph->blockConfig("spec", saved) == Result::SUCCESS);
    REQUIRE(saved.at("axis").type() == typeid(I64));
    REQUIRE(std::any_cast<I64>(saved.at("axis")) == 0);

    const Tensor out = block.outputs.at("buffer").tensor;
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

    engineConfig["axis"] = std::string("-2");
    REQUIRE(flowgraph->blockCreate("spec_too_negative", "spectrum_engine", engineConfig, inputs) ==
            Result::SUCCESS);
    REQUIRE(viewBlock("spec_too_negative").state == Block::State::Errored);
}

TEST_CASE_METHOD(FlowgraphFixture,
                 "Spectrum engine applies window and FFT along a non-last axis",
                 "[modules][dsp][spectrum_engine][block][axis]") {
    Blocks::OnesTensor source;
    source.shape = {4, 3};
    source.dataType = "CF32";
    REQUIRE(flowgraph->blockCreate("src", source, {}) == Result::SUCCESS);

    TensorMap inputs;
    inputs["buffer"].requested("src", "buffer");

    Blocks::SpectrumEngine config;
    config.axis = 0;
    REQUIRE(flowgraph->blockCreate("spec", config, inputs) == Result::SUCCESS);
    REQUIRE(viewBlock("spec").state == Block::State::Created);

    config.axis = -2;
    REQUIRE(flowgraph->blockCreate("spec_negative", config, inputs) == Result::SUCCESS);
    REQUIRE(viewBlock("spec_negative").state == Block::State::Created);
    REQUIRE(flowgraph->compute() == Result::SUCCESS);

    const Tensor positiveOut = viewBlock("spec").outputs.at("buffer").tensor;
    const Tensor negativeOut = viewBlock("spec_negative").outputs.at("buffer").tensor;
    REQUIRE(positiveOut.shape() == Shape{4, 3});
    REQUIRE(negativeOut.shape() == positiveOut.shape());
    REQUIRE(positiveOut.dtype() == DataType::F32);
    REQUIRE(negativeOut.dtype() == positiveOut.dtype());

    const F32 expectedPeak = 20.0f * std::log10((0.42f * 3.0f) / 4.0f);
    for (U64 column = 0; column < positiveOut.shape(1); ++column) {
        REQUIRE_THAT(positiveOut.at<F32>(2, column),
                     Catch::Matchers::WithinAbs(expectedPeak, 0.1f));
        REQUIRE_THAT(negativeOut.at<F32>(2, column),
                     Catch::Matchers::WithinAbs(expectedPeak, 0.1f));
        REQUIRE_THAT(negativeOut.at<F32>(2, column),
                     Catch::Matchers::WithinAbs(positiveOut.at<F32>(2, column), 1e-4f));
    }
}
