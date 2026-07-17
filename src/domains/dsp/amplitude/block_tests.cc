#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>
#include <string>

#include "flowgraph_fixture.hh"
#include "jetstream/domains/core/ones_tensor/block.hh"
#include "jetstream/domains/dsp/amplitude/block.hh"

using namespace Jetstream;

TEST_CASE_METHOD(FlowgraphFixture,
                 "Amplitude block converts CF32 signal to F32",
                 "[modules][dsp][amplitude][block]") {
    Parser::Map sourceConfig;
    sourceConfig["signalDataType"] = std::string("CF32");
    sourceConfig["bufferSize"] = std::string("128");

    REQUIRE(flowgraph->blockCreate("src", "signal_generator", sourceConfig, {}) ==
            Result::SUCCESS);

    TensorMap inputs;
    inputs["signal"].requested("src", "signal");

    REQUIRE(flowgraph->blockCreate("amp", "amplitude", {}, inputs) == Result::SUCCESS);
    REQUIRE(viewBlock("amp").state == Block::State::Created);

    const Tensor out = viewBlock("amp").outputs.at("signal").tensor;
    REQUIRE(out.dtype() == DataType::F32);
    REQUIRE(out.shape(0) == 128);
}

TEST_CASE_METHOD(FlowgraphFixture,
                 "Amplitude block normalizes along the configured axis",
                 "[modules][dsp][amplitude][block][axis]") {
    Blocks::OnesTensor source;
    source.shape = {5, 3};
    source.dataType = "F32";
    REQUIRE(flowgraph->blockCreate("src", source, {}) == Result::SUCCESS);

    TensorMap inputs;
    inputs["signal"].requested("src", "buffer");

    Blocks::Amplitude config;
    config.axis = 0;
    REQUIRE(flowgraph->blockCreate("amp", config, inputs) == Result::SUCCESS);
    REQUIRE(viewBlock("amp").state == Block::State::Created);
    REQUIRE(flowgraph->compute() == Result::SUCCESS);

    const Tensor out = viewBlock("amp").outputs.at("signal").tensor;
    const F32 expected = 20.0f * std::log10(1.0f / 5.0f);
    for (U64 index = 0; index < out.size(); ++index) {
        REQUIRE_THAT(out.data<F32>()[index],
                     Catch::Matchers::WithinAbs(expected, 0.1f));
    }
}
