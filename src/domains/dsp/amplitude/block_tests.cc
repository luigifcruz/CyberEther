#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <algorithm>
#include <cmath>
#include <string>

#include "flowgraph_fixture.hh"
#include "jetstream/domains/core/ones_tensor/block.hh"
#include "jetstream/domains/dsp/amplitude/block.hh"
#include "jetstream/domains/dsp/signal_generator/block.hh"

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

TEST_CASE_METHOD(FlowgraphFixture,
                 "Amplitude block delegates axis validation to its module",
                 "[modules][dsp][amplitude][block][validation]") {
    Blocks::OnesTensor source;
    source.shape = {4};
    source.dataType = "F32";
    REQUIRE(flowgraph->blockCreate("amp_axis_src", source, {}) == Result::SUCCESS);

    TensorMap inputs;
    inputs["signal"].requested("amp_axis_src", "buffer");

    Blocks::Amplitude config;
    config.axis = 1;
    REQUIRE(flowgraph->blockCreate("amp_axis_bad", config, inputs) == Result::SUCCESS);
    REQUIRE(viewBlock("amp_axis_bad").state == Block::State::Errored);
    REQUIRE(viewBlock("amp_axis_bad").outputs.empty());
}

TEST_CASE_METHOD(FlowgraphFixture,
                 "Amplitude block delegates dtype validation to its module",
                 "[modules][dsp][amplitude][block][validation]") {
    Blocks::OnesTensor source;
    source.shape = {4};
    source.dataType = "F64";
    REQUIRE(flowgraph->blockCreate("amp_dtype_src", source, {}) == Result::SUCCESS);

    TensorMap inputs;
    inputs["signal"].requested("amp_dtype_src", "buffer");

    Blocks::Amplitude config;
    REQUIRE(flowgraph->blockCreate("amp_dtype_bad", config, inputs) == Result::SUCCESS);
    REQUIRE(viewBlock("amp_dtype_bad").state == Block::State::Errored);
    REQUIRE(viewBlock("amp_dtype_bad").outputs.empty());
}

TEST_CASE_METHOD(FlowgraphFixture,
                  "Amplitude block preserves invalid axis for recovery",
                 "[modules][dsp][amplitude][block][reconfigure][validation]") {
    Blocks::SignalGenerator source;
    source.signalType = "dc";
    source.bufferSize = 4;
    REQUIRE(flowgraph->blockCreate("amp_update_src", source, {}) == Result::SUCCESS);

    TensorMap inputs;
    inputs["signal"].requested("amp_update_src", "signal");

    Blocks::Amplitude config;
    config.axis = 0;
    REQUIRE(flowgraph->blockCreate("amp_update", config, inputs) == Result::SUCCESS);
    REQUIRE(flowgraph->compute() == Result::SUCCESS);

    Parser::Map update;
    update["axis"] = I64{1};
    REQUIRE(flowgraph->blockReconfigure("amp_update", update) == Result::SUCCESS);
    REQUIRE(viewBlock("amp_update").state == Block::State::Errored);
    REQUIRE(viewBlock("amp_update").outputs.empty());

    Parser::Map saved;
    REQUIRE(flowgraph->blockConfig("amp_update", saved) == Result::SUCCESS);
    REQUIRE(std::any_cast<I64>(saved.at("axis")) == 1);

    Parser::Map recovery;
    recovery["axis"] = config.axis;
    REQUIRE(flowgraph->blockReconfigure("amp_update", recovery) == Result::SUCCESS);
    REQUIRE(viewBlock("amp_update").state == Block::State::Created);

    Tensor output = viewBlock("amp_update").outputs.at("signal").tensor;
    std::fill(output.data<F32>(), output.data<F32>() + output.size(), 12345.0f);

    REQUIRE(flowgraph->compute() == Result::SUCCESS);
    const F32 expected = 20.0f * std::log10(0.25f);
    for (U64 index = 0; index < output.size(); ++index) {
        REQUIRE_THAT(output.at<F32>(index),
                     Catch::Matchers::WithinAbs(expected, 0.1f));
    }
}
