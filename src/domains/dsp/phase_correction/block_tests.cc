#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "flowgraph_fixture.hh"
#include "jetstream/domains/dsp/phase_correction/block.hh"
#include "jetstream/domains/dsp/signal_generator/block.hh"

using namespace Jetstream;

TEST_CASE_METHOD(FlowgraphFixture,
                 "Phase correction block preserves phase across submissions",
                 "[modules][dsp][phase_correction][block]") {
    Blocks::SignalGenerator source;
    source.signalType = "dc";
    source.signalDataType = "CF32";
    source.bufferSize = 4;
    REQUIRE(flowgraph->blockCreate("src", source, {}) == Result::SUCCESS);

    TensorMap inputs;
    inputs["signal"].requested("src", "signal");

    Blocks::PhaseCorrection config;
    config.phaseIncrement = JST_PI / 2.0;
    REQUIRE(flowgraph->blockCreate("phase", config, inputs) == Result::SUCCESS);
    REQUIRE(viewBlock("phase").state == Block::State::Created);

    Parser::Map saved;
    REQUIRE(flowgraph->blockConfig("phase", saved) == Result::SUCCESS);
    REQUIRE(saved.at("phaseIncrement").type() == typeid(F64));
    REQUIRE(std::any_cast<F64>(saved.at("phaseIncrement")) == config.phaseIncrement);

    const Tensor output = viewBlock("phase").outputs.at("signal").tensor;
    REQUIRE(output.dtype() == DataType::CF32);
    REQUIRE(output.shape() == Shape{4});

    REQUIRE(flowgraph->compute() == Result::SUCCESS);
    for (U64 index = 0; index < output.size(); ++index) {
        REQUIRE_THAT(output.at<CF32>(index).real(),
                     Catch::Matchers::WithinAbs(1.0f, 1e-5f));
        REQUIRE_THAT(output.at<CF32>(index).imag(),
                     Catch::Matchers::WithinAbs(0.0f, 1e-5f));
    }

    REQUIRE(flowgraph->compute() == Result::SUCCESS);
    for (U64 index = 0; index < output.size(); ++index) {
        REQUIRE_THAT(output.at<CF32>(index).real(),
                     Catch::Matchers::WithinAbs(0.0f, 1e-5f));
        REQUIRE_THAT(output.at<CF32>(index).imag(),
                     Catch::Matchers::WithinAbs(1.0f, 1e-5f));
    }
}

TEST_CASE_METHOD(FlowgraphFixture,
                 "Phase correction block requires a signal input",
                 "[modules][dsp][phase_correction][block][validation]") {
    Blocks::PhaseCorrection config;
    const auto result = flowgraph->blockCreate("phase", config, {});
    REQUIRE((result == Result::SUCCESS || result == Result::INCOMPLETE));
    REQUIRE(viewBlock("phase").state == Block::State::Incomplete);
}
