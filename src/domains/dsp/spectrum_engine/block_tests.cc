#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <algorithm>
#include <cmath>
#include <string>
#include <unordered_map>
#include <vector>

#include "flowgraph_fixture.hh"
#include "jetstream/domains/core/ones_tensor/block.hh"
#include "jetstream/domains/dsp/spectrum_engine/block.hh"
#include "jetstream/registry.hh"

using namespace Jetstream;

TEST_CASE("Spectrum engine declares conditional module requirements",
          "[modules][dsp][spectrum_engine][requirements]") {
    const std::vector<Registry::BlockModuleRequirement> expected = {
        {"cast"},
        {"window"},
        {"invert"},
        {"reshape"},
        {"multiply"},
        {"fft"},
        {"amplitude"},
        {"agc", true},
        {"range", true},
    };

    const auto registrations = Registry::ListAvailableBlocks("spectrum_engine");
    REQUIRE(registrations.size() == 1);
    REQUIRE(registrations.front().moduleRequirements == expected);
}

TEST_CASE("Spectrum engine targets require only unconditional modules",
          "[modules][dsp][spectrum_engine][requirements]") {
    const auto targets = Registry::ListAvailableBlockTargets("spectrum_engine");
    const auto cpu = std::find_if(
        targets.begin(), targets.end(), [](const auto& target) {
            return target.device == DeviceType::CPU &&
                   target.runtime == RuntimeType::NATIVE &&
                   target.provider == "generic";
        });
    REQUIRE(cpu != targets.end());

    const auto cudaWindow = Registry::ListAvailableModules("window",
                                                            DeviceType::CUDA,
                                                            RuntimeType::NATIVE,
                                                            "generic");
    if (!cudaWindow.empty()) {
        const auto cuda = std::find_if(
            targets.begin(), targets.end(), [](const auto& target) {
                return target.device == DeviceType::CUDA &&
                       target.runtime == RuntimeType::NATIVE &&
                       target.provider == "generic";
            });
        REQUIRE(cuda != targets.end());
    }
}

TEST_CASE_METHOD(FlowgraphFixture,
                 "Spectrum engine block creates and exposes scaled spectrum",
                 "[modules][dsp][spectrum_engine][block]") {
    Parser::Map sourceConfig;
    sourceConfig["signalDataType"] = std::string("F32");
    sourceConfig["bufferSize"] = std::string("256");

    REQUIRE(flowgraph->blockCreate("src", "signal_generator", sourceConfig, {}) ==
            Result::SUCCESS);
    Tensor source = viewBlock("src").outputs.at("signal").tensor;
    REQUIRE(source.setAttribute("sampleAxis", Index{0}) == Result::SUCCESS);

    Parser::Map engineConfig;
    engineConfig["enableScale"] = std::string("true");
    engineConfig["rangeMin"] = std::string("-100");
    engineConfig["rangeMax"] = std::string("0");

    TensorMap inputs;
    inputs["buffer"].requested("src", "signal");

    REQUIRE(flowgraph->blockCreate("spec", "spectrum_engine", engineConfig, inputs) ==
            Result::SUCCESS);
    const auto block = viewBlock("spec");
    REQUIRE(block.state == Block::State::Created);

    const Tensor out = block.outputs.at("buffer").tensor;
    REQUIRE(out.dtype() == DataType::F32);
    REQUIRE(out.shape(0) == 256);
    REQUIRE(std::any_cast<Index>(out.attribute("sampleAxis")) == 0);
    REQUIRE_FALSE(out.hasAttribute("batchAxis"));
    REQUIRE_FALSE(out.hasAttribute("channelAxis"));

    for (U64 cycle = 0; cycle < 3; ++cycle) {
        REQUIRE(flowgraph->compute() == Result::SUCCESS);
    }

    std::vector<Flowgraph::View::MetricEntry> sourceMetrics;
    REQUIRE(flowgraph->view().metrics("src", sourceMetrics) == Result::SUCCESS);
    REQUIRE(sourceMetrics.size() == 1);
    const auto* sourceTiming = std::any_cast<Module::Timing>(&sourceMetrics.front().value);
    REQUIRE(sourceTiming != nullptr);
    REQUIRE(sourceTiming->cycles == 3);

    std::vector<Flowgraph::View::MetricEntry> engineMetrics;
    REQUIRE(flowgraph->view().metrics("spec", engineMetrics) == Result::SUCCESS);

    std::unordered_map<std::string, U64> cycles;
    std::unordered_map<std::string, F32> computeTimes;
    for (const auto& metric : engineMetrics) {
        const auto* timing = std::any_cast<Module::Timing>(&metric.value);
        REQUIRE(timing != nullptr);
        cycles[metric.name] = timing->cycles;
        computeTimes[metric.name] = timing->computeTime;
    }

    REQUIRE(cycles.at("runtime:window") == 1);
    REQUIRE(cycles.at("runtime:invert") == 1);
    REQUIRE(cycles.at("runtime:multiply") == 3);
    REQUIRE(cycles.at("runtime:fft") == 3);
    REQUIRE(computeTimes.at("runtime:window") == 0.0f);
    REQUIRE(computeTimes.at("runtime:invert") == 0.0f);
}

TEST_CASE_METHOD(FlowgraphFixture,
                  "Spectrum engine centers odd lengths on an FFT bin",
                  "[modules][dsp][spectrum_engine][block]") {
    Blocks::OnesTensor source;
    source.shape = {5};
    source.dataType = "CF32";
    REQUIRE(flowgraph->blockCreate("odd_src", source, {}) == Result::SUCCESS);

    Tensor sourceTensor = viewBlock("odd_src").outputs.at("buffer").tensor;
    REQUIRE(sourceTensor.setAttribute("sampleAxis", Index{0}) == Result::SUCCESS);

    TensorMap inputs;
    inputs["buffer"].requested("odd_src", "buffer");
    REQUIRE(flowgraph->blockCreate("odd_spec", Blocks::SpectrumEngine{}, inputs) ==
            Result::SUCCESS);
    REQUIRE(flowgraph->compute() == Result::SUCCESS);

    const Tensor output = viewBlock("odd_spec").outputs.at("buffer").tensor;
    const U64 center = output.shape(0) / 2;
    const F32 expectedPeak = 20.0f * std::log10((0.42f * 4.0f) / 5.0f);
    REQUIRE_THAT(output.at<F32>(center),
                 Catch::Matchers::WithinAbs(expectedPeak, 0.1f));

    const auto peak = std::max_element(output.data<F32>(),
                                       output.data<F32>() + output.size());
    REQUIRE(static_cast<U64>(peak - output.data<F32>()) == center);
}

TEST_CASE_METHOD(FlowgraphFixture,
                 "Spectrum engine executes the optional AGC path",
                 "[modules][dsp][spectrum_engine][block][agc]") {
    Blocks::OnesTensor source;
    source.shape = {2048};
    source.dataType = "CF32";
    REQUIRE(flowgraph->blockCreate("agc_src", source, {}) == Result::SUCCESS);

    Tensor sourceTensor = viewBlock("agc_src").outputs.at("buffer").tensor;
    REQUIRE(sourceTensor.setAttribute("sampleAxis", Index{0}) == Result::SUCCESS);

    TensorMap inputs;
    inputs["buffer"].requested("agc_src", "buffer");
    Blocks::SpectrumEngine config;
    config.enableAgc = true;
    REQUIRE(flowgraph->blockCreate("agc_spec", config, inputs) == Result::SUCCESS);
    REQUIRE(flowgraph->compute() == Result::SUCCESS);

    const Tensor output = viewBlock("agc_spec").outputs.at("buffer").tensor;
    REQUIRE(output.dtype() == DataType::F32);
    REQUIRE(output.shape() == source.shape);
    for (U64 i = 0; i < output.size(); ++i) {
        REQUIRE_FALSE(std::isnan(output.at<F32>(i)));
    }
}

TEST_CASE_METHOD(FlowgraphFixture,
                  "Spectrum engine block rejects invalid signal metadata",
                  "[modules][dsp][spectrum_engine][block][validation]") {
    Blocks::OnesTensor source;
    source.shape = {8, 6};
    source.dataType = "CF32";

    SECTION("batch axis has the wrong type") {
        REQUIRE(flowgraph->blockCreate("src", source, {}) == Result::SUCCESS);
        Tensor tensor = viewBlock("src").outputs.at("buffer").tensor;
        REQUIRE(tensor.setAttribute("sampleAxis", Index{1}) == Result::SUCCESS);
        REQUIRE(tensor.setAttribute("batchAxis", I64{0}) == Result::SUCCESS);
    }
    SECTION("batch axis is out of bounds") {
        REQUIRE(flowgraph->blockCreate("src", source, {}) == Result::SUCCESS);
        Tensor tensor = viewBlock("src").outputs.at("buffer").tensor;
        REQUIRE(tensor.setAttribute("sampleAxis", Index{1}) == Result::SUCCESS);
        REQUIRE(tensor.setAttribute("batchAxis", Index{2}) == Result::SUCCESS);
    }
    SECTION("sample axis is missing") {
        REQUIRE(flowgraph->blockCreate("src", source, {}) == Result::SUCCESS);
    }
    SECTION("sample axis has the wrong type") {
        REQUIRE(flowgraph->blockCreate("src", source, {}) == Result::SUCCESS);
        Tensor tensor = viewBlock("src").outputs.at("buffer").tensor;
        REQUIRE(tensor.setAttribute("sampleAxis", I64{1}) == Result::SUCCESS);
    }
    SECTION("sample axis is out of bounds") {
        REQUIRE(flowgraph->blockCreate("src", source, {}) == Result::SUCCESS);
        Tensor tensor = viewBlock("src").outputs.at("buffer").tensor;
        REQUIRE(tensor.setAttribute("sampleAxis", Index{2}) == Result::SUCCESS);
    }
    SECTION("signal roles share an axis") {
        REQUIRE(flowgraph->blockCreate("src", source, {}) == Result::SUCCESS);
        Tensor tensor = viewBlock("src").outputs.at("buffer").tensor;
        REQUIRE(tensor.setAttribute("sampleAxis", Index{1}) == Result::SUCCESS);
        REQUIRE(tensor.setAttribute("channelAxis", Index{1}) == Result::SUCCESS);
    }

    TensorMap inputs;
    inputs["buffer"].requested("src", "buffer");

    REQUIRE(flowgraph->blockCreate("spec_bad", Blocks::SpectrumEngine{}, inputs) ==
            Result::SUCCESS);
    const auto block = viewBlock("spec_bad");
    REQUIRE(block.state == Block::State::Errored);
    REQUIRE(block.outputs.empty());
    REQUIRE(block.diagnostic.find("[BLOCK_SPECTRUM_ENGINE]") != std::string::npos);
}

TEST_CASE_METHOD(FlowgraphFixture,
                  "Spectrum engine remains incomplete without a resolved input",
                  "[modules][dsp][spectrum_engine][block][validation]") {
    Blocks::SpectrumEngine config;

    SECTION("absent input") {
        REQUIRE(flowgraph->blockCreate("spec_absent", config, {}) == Result::SUCCESS);
        REQUIRE(viewBlock("spec_absent").state == Block::State::Incomplete);
        REQUIRE(viewBlock("spec_absent").outputs.empty());
    }

    SECTION("unresolved input") {
        Blocks::OnesTensor source;
        source.shape = {4};
        source.dataType = "CF32";
        REQUIRE(flowgraph->blockCreate("spec_unresolved_src", source, {}) ==
                Result::SUCCESS);

        TensorMap inputs;
        inputs["buffer"].requested("spec_unresolved_src", "missing");

        REQUIRE(flowgraph->blockCreate("spec_unresolved", config, inputs) ==
                Result::SUCCESS);
        REQUIRE(viewBlock("spec_unresolved").state == Block::State::Incomplete);
        REQUIRE(viewBlock("spec_unresolved").outputs.empty());
    }
}

TEST_CASE_METHOD(FlowgraphFixture,
                  "Spectrum engine recreates for topology and range changes",
                  "[modules][dsp][spectrum_engine][block][reconfigure]") {
    Blocks::OnesTensor source;
    source.shape = {4, 3};
    source.dataType = "CF32";
    REQUIRE(flowgraph->blockCreate("spec_recreate_src", source, {}) == Result::SUCCESS);
    Tensor sourceTensor = viewBlock("spec_recreate_src").outputs.at("buffer").tensor;
    REQUIRE(sourceTensor.setAttribute("sampleAxis", Index{1}) == Result::SUCCESS);
    REQUIRE(sourceTensor.setAttribute("batchAxis", Index{0}) == Result::SUCCESS);

    TensorMap inputs;
    inputs["buffer"].requested("spec_recreate_src", "buffer");

    Blocks::SpectrumEngine config;
    REQUIRE(flowgraph->blockCreate("spec_recreate", config, inputs) == Result::SUCCESS);
    const auto originalOutputId =
        viewBlock("spec_recreate").outputs.at("buffer").tensor.id();

    Parser::Map update;
    update["enableScale"] = true;
    update["rangeMin"] = -100.0f;
    update["rangeMax"] = -10.0f;
    REQUIRE(flowgraph->blockReconfigure("spec_recreate", update) == Result::SUCCESS);

    const auto recreated = viewBlock("spec_recreate");
    REQUIRE(recreated.state == Block::State::Created);
    REQUIRE(recreated.outputs.at("buffer").tensor.id() != originalOutputId);
    REQUIRE(recreated.outputs.at("buffer").tensor.shape() == Shape{4, 3});
    REQUIRE(recreated.outputs.at("buffer").tensor.dtype() == DataType::F32);
    REQUIRE(recreated.outputs.at("buffer").tensor.attribute("batchAxis").type() ==
            typeid(Index));
    REQUIRE(std::any_cast<Index>(
                recreated.outputs.at("buffer").tensor.attribute("batchAxis")) == 0);
    REQUIRE(std::any_cast<Index>(
                recreated.outputs.at("buffer").tensor.attribute("sampleAxis")) == 1);

    Parser::Map savedMap;
    REQUIRE(flowgraph->blockConfig("spec_recreate", savedMap) == Result::SUCCESS);
    Blocks::SpectrumEngine saved;
    REQUIRE(saved.deserialize(savedMap) == Result::SUCCESS);
    REQUIRE(saved.enableScale);
    REQUIRE(saved.rangeMin == -100.0f);
    REQUIRE(saved.rangeMax == -10.0f);

    Tensor output = recreated.outputs.at("buffer").tensor;
    std::fill(output.data<F32>(), output.data<F32>() + output.size(), 12345.0f);
    REQUIRE(flowgraph->compute() == Result::SUCCESS);
    REQUIRE(std::any_of(output.data<F32>(), output.data<F32>() + output.size(),
                        [](const F32 value) { return value != 12345.0f; }));
}

TEST_CASE_METHOD(FlowgraphFixture,
                 "Spectrum engine delegates dtype validation to child modules",
                 "[modules][dsp][spectrum_engine][block][validation]") {
    Blocks::OnesTensor source;
    source.shape = {4};
    source.dataType = "F64";
    REQUIRE(flowgraph->blockCreate("spec_dtype_src", source, {}) == Result::SUCCESS);
    Tensor sourceTensor = viewBlock("spec_dtype_src").outputs.at("buffer").tensor;
    REQUIRE(sourceTensor.setAttribute("sampleAxis", Index{0}) == Result::SUCCESS);

    TensorMap inputs;
    inputs["buffer"].requested("spec_dtype_src", "buffer");

    Blocks::SpectrumEngine config;
    config.enableScale = true;
    REQUIRE(flowgraph->blockCreate("spec_dtype", config, inputs) == Result::SUCCESS);
    const auto block = viewBlock("spec_dtype");
    REQUIRE(block.state == Block::State::Errored);
    REQUIRE(block.outputs.empty());
    REQUIRE(std::any_of(block.interfaceConfigs.begin(),
                        block.interfaceConfigs.end(),
                        [](const auto& field) { return field.name == "rangeMin"; }));
    REQUIRE(std::any_of(block.interfaceConfigs.begin(),
                        block.interfaceConfigs.end(),
                        [](const auto& field) { return field.name == "rangeMax"; }));
}

TEST_CASE_METHOD(FlowgraphFixture,
                  "Spectrum engine preserves channel and batch roles",
                  "[modules][dsp][spectrum_engine][block][metadata]") {
    Blocks::OnesTensor source;
    source.dataType = "CF32";

    bool batched = false;
    SECTION("channels and samples") {
        source.shape = {3, 4};
    }
    SECTION("opaque axis, channels, and samples") {
        source.shape = {2, 3, 4};
    }
    SECTION("batch, heads, and samples") {
        source.shape = {2, 3, 4};
        batched = true;
    }

    REQUIRE(flowgraph->blockCreate("heads_src", source, {}) == Result::SUCCESS);
    Tensor sourceTensor = viewBlock("heads_src").outputs.at("buffer").tensor;
    const Index sampleAxis = source.shape.size() - 1;
    const Index channelAxis = source.shape.size() - 2;
    REQUIRE(sourceTensor.setAttribute("sampleAxis", sampleAxis) == Result::SUCCESS);
    REQUIRE(sourceTensor.setAttribute("channelAxis", channelAxis) == Result::SUCCESS);
    if (batched) {
        REQUIRE(sourceTensor.setAttribute("batchAxis", Index{0}) == Result::SUCCESS);
    }

    TensorMap inputs;
    inputs["buffer"].requested("heads_src", "buffer");
    REQUIRE(flowgraph->blockCreate("heads_spec", Blocks::SpectrumEngine{}, inputs) ==
            Result::SUCCESS);
    REQUIRE(flowgraph->compute() == Result::SUCCESS);

    const Tensor output = viewBlock("heads_spec").outputs.at("buffer").tensor;
    REQUIRE(output.shape() == source.shape);
    REQUIRE(output.dtype() == DataType::F32);
    REQUIRE(std::any_cast<Index>(output.attribute("sampleAxis")) == sampleAxis);
    REQUIRE(std::any_cast<Index>(output.attribute("channelAxis")) == channelAxis);
    if (batched) {
        REQUIRE(output.attribute("batchAxis").type() == typeid(Index));
        REQUIRE(std::any_cast<Index>(output.attribute("batchAxis")) == 0);
    } else {
        REQUIRE_FALSE(output.hasAttribute("batchAxis"));
    }

    const F32 expectedPeak = 20.0f * std::log10((0.42f * 3.0f) / 4.0f);
    for (U64 lane = 0; lane < output.size() / 4; ++lane) {
        REQUIRE_THAT(output.data<F32>()[lane * 4 + 2],
                     Catch::Matchers::WithinAbs(expectedPeak, 0.1f));
    }
}

TEST_CASE_METHOD(FlowgraphFixture,
                  "Spectrum engine uses explicit samples around leading and trailing batches",
                  "[modules][dsp][spectrum_engine][block][metadata]") {
    Blocks::OnesTensor source;
    source.dataType = "CF32";

    Index batchAxis = 0;
    Shape expectedShape;
    SECTION("leading batch axis") {
        source.shape = {3, 4};
        batchAxis = 0;
        expectedShape = {3, 4};
    }
    SECTION("trailing batch axis") {
        source.shape = {4, 3};
        batchAxis = 1;
        expectedShape = {4, 3};
    }

    REQUIRE(flowgraph->blockCreate("src", source, {}) == Result::SUCCESS);
    Tensor sourceTensor = viewBlock("src").outputs.at("buffer").tensor;
    const Index sampleAxis = batchAxis == 0 ? Index{1} : Index{0};
    REQUIRE(sourceTensor.setAttribute("sampleAxis", sampleAxis) == Result::SUCCESS);
    REQUIRE(sourceTensor.setAttribute("batchAxis", batchAxis) == Result::SUCCESS);

    TensorMap inputs;
    inputs["buffer"].requested("src", "buffer");

    REQUIRE(flowgraph->blockCreate("spec", Blocks::SpectrumEngine{}, inputs) ==
            Result::SUCCESS);
    REQUIRE(viewBlock("spec").state == Block::State::Created);
    REQUIRE(flowgraph->compute() == Result::SUCCESS);

    const Tensor output = viewBlock("spec").outputs.at("buffer").tensor;
    REQUIRE(output.shape() == expectedShape);
    REQUIRE(output.dtype() == DataType::F32);
    REQUIRE(std::any_cast<Index>(output.attribute("sampleAxis")) == sampleAxis);
    REQUIRE(output.attribute("batchAxis").type() == typeid(Index));
    REQUIRE(std::any_cast<Index>(output.attribute("batchAxis")) == batchAxis);

    const F32 expectedPeak = 20.0f * std::log10((0.42f * 3.0f) / 4.0f);
    for (U64 batch = 0; batch < output.shape(batchAxis); ++batch) {
        const F32 peak = batchAxis == 0 ? output.at<F32>(batch, 2)
                                        : output.at<F32>(2, batch);
        REQUIRE_THAT(peak, Catch::Matchers::WithinAbs(expectedPeak, 0.1f));
    }
}
