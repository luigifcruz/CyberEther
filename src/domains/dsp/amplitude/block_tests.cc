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

    Tensor sourceTensor = viewBlock("src").outputs.at("signal").tensor;
    REQUIRE(sourceTensor.setAttribute("sampleAxis", Index{0}) == Result::SUCCESS);

    TensorMap inputs;
    inputs["signal"].requested("src", "signal");

    REQUIRE(flowgraph->blockCreate("amp", "amplitude", {}, inputs) == Result::SUCCESS);
    REQUIRE(viewBlock("amp").state == Block::State::Created);

    const Tensor out = viewBlock("amp").outputs.at("signal").tensor;
    REQUIRE(out.dtype() == DataType::F32);
    REQUIRE(out.shape(0) == 128);
}

TEST_CASE_METHOD(FlowgraphFixture,
                  "Amplitude block uses explicit samples before a trailing batch",
                 "[modules][dsp][amplitude][block][batch][metadata]") {
    Blocks::OnesTensor source;
    source.shape = {5, 3};
    source.dataType = "F32";
    REQUIRE(flowgraph->blockCreate("src", source, {}) == Result::SUCCESS);

    Tensor sourceTensor = viewBlock("src").outputs.at("buffer").tensor;
    REQUIRE(sourceTensor.setAttribute("sampleAxis", Index{0}) == Result::SUCCESS);
    REQUIRE(sourceTensor.setAttribute("batchAxis", Index{1}) == Result::SUCCESS);

    TensorMap inputs;
    inputs["signal"].requested("src", "buffer");

    Blocks::Amplitude config;
    REQUIRE(flowgraph->blockCreate("amp", config, inputs) == Result::SUCCESS);
    REQUIRE(viewBlock("amp").state == Block::State::Created);
    REQUIRE(flowgraph->compute() == Result::SUCCESS);

    const Tensor out = viewBlock("amp").outputs.at("signal").tensor;
    REQUIRE(std::any_cast<Index>(out.attribute("sampleAxis")) == Index{0});
    REQUIRE(out.hasAttribute("batchAxis"));
    REQUIRE(std::any_cast<Index>(out.attribute("batchAxis")) == Index{1});
    const F32 expected = 20.0f * std::log10(1.0f / 5.0f);
    for (U64 index = 0; index < out.size(); ++index) {
        REQUIRE_THAT(out.data<F32>()[index],
                     Catch::Matchers::WithinAbs(expected, 0.1f));
    }
}

TEST_CASE_METHOD(FlowgraphFixture,
                   "Amplitude block normalizes multi-head samples",
                  "[modules][dsp][amplitude][block][heads]") {
    Blocks::OnesTensor source;
    source.shape = {2, 4};
    source.dataType = "F32";
    REQUIRE(flowgraph->blockCreate("amp_heads_src", source, {}) == Result::SUCCESS);

    Tensor sourceTensor = viewBlock("amp_heads_src").outputs.at("buffer").tensor;
    REQUIRE(sourceTensor.setAttribute("sampleAxis", Index{1}) == Result::SUCCESS);
    REQUIRE(sourceTensor.setAttribute("channelAxis", Index{0}) == Result::SUCCESS);

    TensorMap inputs;
    inputs["signal"].requested("amp_heads_src", "buffer");

    Blocks::Amplitude config;
    REQUIRE(flowgraph->blockCreate("amp_heads", config, inputs) == Result::SUCCESS);
    REQUIRE(viewBlock("amp_heads").state == Block::State::Created);
    REQUIRE(flowgraph->compute() == Result::SUCCESS);

    const Tensor output = viewBlock("amp_heads").outputs.at("signal").tensor;
    REQUIRE(output.shape() == Shape{2, 4});
    REQUIRE(std::any_cast<Index>(output.attribute("sampleAxis")) == Index{1});
    REQUIRE(std::any_cast<Index>(output.attribute("channelAxis")) == Index{0});
    REQUIRE_FALSE(output.hasAttribute("batchAxis"));
    const F32 expected = 20.0f * std::log10(0.25f);
    for (U64 index = 0; index < output.size(); ++index) {
        REQUIRE_THAT(output.at<F32>(index),
                     Catch::Matchers::WithinAbs(expected, 0.1f));
    }
}

TEST_CASE_METHOD(FlowgraphFixture,
                 "Amplitude block delegates dtype validation to its module",
                 "[modules][dsp][amplitude][block][validation]") {
    Blocks::OnesTensor source;
    source.shape = {4};
    source.dataType = "F64";
    REQUIRE(flowgraph->blockCreate("amp_dtype_src", source, {}) == Result::SUCCESS);

    Tensor sourceTensor = viewBlock("amp_dtype_src").outputs.at("buffer").tensor;
    REQUIRE(sourceTensor.setAttribute("sampleAxis", Index{0}) == Result::SUCCESS);

    TensorMap inputs;
    inputs["signal"].requested("amp_dtype_src", "buffer");

    Blocks::Amplitude config;
    REQUIRE(flowgraph->blockCreate("amp_dtype_bad", config, inputs) == Result::SUCCESS);
    REQUIRE(viewBlock("amp_dtype_bad").state == Block::State::Errored);
    REQUIRE(viewBlock("amp_dtype_bad").outputs.empty());
}
