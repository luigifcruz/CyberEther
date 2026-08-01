#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <algorithm>
#include <string>

#include "flowgraph_fixture.hh"
#include "jetstream/domains/core/ones_tensor/block.hh"
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

    Tensor sourceTensor = viewBlock("src").outputs.at("signal").tensor;
    REQUIRE(sourceTensor.setAttribute("sampleAxis", Index{0}) == Result::SUCCESS);

    TensorMap inputs;
    inputs["signal"].requested("src", "signal");

    REQUIRE(flowgraph->blockCreate("fft", "fft", {}, inputs) == Result::SUCCESS);
    REQUIRE(viewBlock("fft").state == Block::State::Created);

    const Tensor out = viewBlock("fft").outputs.at("signal").tensor;
    REQUIRE(out.dtype() == DataType::CF32);
    REQUIRE(out.shape(0) == 64);

    const auto& interfaceConfigs = viewBlock("fft").interfaceConfigs;
    const auto complexOutput = std::find_if(
        interfaceConfigs.begin(),
        interfaceConfigs.end(),
        [](const auto& field) { return field.name == "complexOutput"; });
    REQUIRE(complexOutput == interfaceConfigs.end());
}

TEST_CASE_METHOD(FlowgraphFixture,
                  "FFT block accepts batched heads and propagates metadata",
                  "[modules][dsp][fft][block][batch][metadata]") {
    Blocks::OnesTensor source;
    source.shape = {2, 3, 4};
    source.dataType = "CF32";
    REQUIRE(flowgraph->blockCreate("fft_batch_src", source, {}) == Result::SUCCESS);

    Tensor sourceTensor = viewBlock("fft_batch_src").outputs.at("buffer").tensor;
    REQUIRE(sourceTensor.setAttribute("sampleAxis", Index{2}) == Result::SUCCESS);
    REQUIRE(sourceTensor.setAttribute("batchAxis", Index{0}) == Result::SUCCESS);
    REQUIRE(sourceTensor.setAttribute("channelAxis", Index{1}) == Result::SUCCESS);

    TensorMap inputs;
    inputs["signal"].requested("fft_batch_src", "buffer");
    REQUIRE(flowgraph->blockCreate("fft_batch", Blocks::Fft{}, inputs) ==
            Result::SUCCESS);
    REQUIRE(viewBlock("fft_batch").state == Block::State::Created);

    const Tensor output = viewBlock("fft_batch").outputs.at("signal").tensor;
    REQUIRE(output.shape() == Shape{2, 3, 4});
    REQUIRE(std::any_cast<Index>(output.attribute("sampleAxis")) == Index{2});
    REQUIRE(output.hasAttribute("batchAxis"));
    REQUIRE(std::any_cast<Index>(output.attribute("batchAxis")) == Index{0});
    REQUIRE(std::any_cast<Index>(output.attribute("channelAxis")) == Index{1});
}

TEST_CASE_METHOD(FlowgraphFixture,
                 "FFT block delegates dtype validation to its module",
                 "[modules][dsp][fft][block][validation]") {
    Blocks::OnesTensor source;
    source.shape = {4};
    source.dataType = "F64";
    REQUIRE(flowgraph->blockCreate("fft_dtype_src", source, {}) == Result::SUCCESS);

    Tensor sourceTensor = viewBlock("fft_dtype_src").outputs.at("buffer").tensor;
    REQUIRE(sourceTensor.setAttribute("sampleAxis", Index{0}) == Result::SUCCESS);

    TensorMap inputs;
    inputs["signal"].requested("fft_dtype_src", "buffer");

    REQUIRE(flowgraph->blockCreate("fft_dtype", Blocks::Fft{}, inputs) ==
            Result::SUCCESS);
    REQUIRE(viewBlock("fft_dtype").state == Block::State::Errored);
    REQUIRE(viewBlock("fft_dtype").outputs.empty());
}

TEST_CASE_METHOD(FlowgraphFixture,
                  "FFT block uses candidate direction before metadata validation",
                  "[modules][dsp][fft][block][validation][interface]") {
    Blocks::OnesTensor source;
    source.shape = {2, 4};
    source.dataType = "F32";
    REQUIRE(flowgraph->blockCreate("fft_inverse_src", source, {}) == Result::SUCCESS);

    Tensor sourceTensor = viewBlock("fft_inverse_src").outputs.at("buffer").tensor;
    REQUIRE(sourceTensor.setAttribute("sampleAxis", Index{1}) == Result::SUCCESS);
    REQUIRE(sourceTensor.setAttribute("batchAxis", I64{0}) == Result::SUCCESS);

    TensorMap inputs;
    inputs["signal"].requested("fft_inverse_src", "buffer");

    Blocks::Fft config;
    config.forward = false;
    REQUIRE(flowgraph->blockCreate("fft_inverse_bad", config, inputs) ==
            Result::SUCCESS);

    const auto block = viewBlock("fft_inverse_bad");
    REQUIRE(block.state == Block::State::Errored);
    const auto complexOutput = std::find_if(
        block.interfaceConfigs.begin(),
        block.interfaceConfigs.end(),
        [](const auto& field) { return field.name == "complexOutput"; });
    REQUIRE(complexOutput == block.interfaceConfigs.end());
}

TEST_CASE_METHOD(FlowgraphFixture,
                 "FFT block exposes complex output for real input",
                 "[modules][dsp][fft][block][real][complex]") {
    Blocks::OnesTensor source;
    source.shape = {4};
    source.dataType = "F32";
    REQUIRE(flowgraph->blockCreate("fft_real_src", source, {}) == Result::SUCCESS);

    Tensor sourceTensor = viewBlock("fft_real_src").outputs.at("buffer").tensor;
    REQUIRE(sourceTensor.setAttribute("sampleAxis", Index{0}) == Result::SUCCESS);

    Blocks::Fft config;
    config.complexOutput = true;

    TensorMap inputs;
    inputs["signal"].requested("fft_real_src", "buffer");
    REQUIRE(flowgraph->blockCreate("fft_real", config, inputs) == Result::SUCCESS);
    const auto block = viewBlock("fft_real");
    REQUIRE(block.state == Block::State::Created);
    const auto complexOutput = std::find_if(
        block.interfaceConfigs.begin(),
        block.interfaceConfigs.end(),
        [](const auto& field) { return field.name == "complexOutput"; });
    REQUIRE(complexOutput != block.interfaceConfigs.end());

    Tensor output = viewBlock("fft_real").outputs.at("signal").tensor;
    REQUIRE(output.dtype() == DataType::CF32);
    REQUIRE(output.shape() == Shape{3});
    REQUIRE(std::any_cast<Index>(output.attribute("sampleAxis")) == Index{0});

    REQUIRE(flowgraph->compute() == Result::SUCCESS);
    REQUIRE_THAT(output.at<CF32>(0).real(),
                 Catch::Matchers::WithinAbs(4.0f, 1e-4f));
    REQUIRE_THAT(output.at<CF32>(0).imag(),
                 Catch::Matchers::WithinAbs(0.0f, 1e-4f));
    for (U64 index = 1; index < output.size(); ++index) {
        REQUIRE_THAT(std::abs(output.at<CF32>(index)),
                     Catch::Matchers::WithinAbs(0.0f, 1e-4f));
    }

    Blocks::Fft inverseConfig;
    inverseConfig.forward = false;
    REQUIRE(flowgraph->blockCreate("fft_real_inverse", inverseConfig, inputs) ==
            Result::SUCCESS);

    const auto inverse = viewBlock("fft_real_inverse");
    const auto inverseComplexOutput = std::find_if(
        inverse.interfaceConfigs.begin(),
        inverse.interfaceConfigs.end(),
        [](const auto& field) { return field.name == "complexOutput"; });
    REQUIRE(inverseComplexOutput == inverse.interfaceConfigs.end());
    REQUIRE(inverse.outputs.at("signal").tensor.dtype() == DataType::F32);
}
