#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <algorithm>
#include <any>
#include <limits>
#include <optional>
#include <string>
#include <vector>

#include "jetstream/domains/dsp/filter_engine/block.hh"
#include "jetstream/domains/dsp/filter_taps/block.hh"
#include "jetstream/domains/core/cast/block.hh"
#include "jetstream/domains/core/ones_tensor/block.hh"
#include "flowgraph_fixture.hh"

using namespace Jetstream;

namespace {

void RequireErroredWithInterface(const Flowgraph::View::BlockData& block) {
    REQUIRE(block.state == Block::State::Errored);
    REQUIRE(block.interfaceInputs.size() == 2);
    REQUIRE(block.interfaceOutputs.size() == 1);
    REQUIRE(block.interfaceConfigs.empty());
    REQUIRE(block.outputs.empty());
}

void TagSignal(Tensor tensor,
               const Index sampleAxis,
               const std::optional<Index> batchAxis = std::nullopt,
               const std::optional<Index> channelAxis = std::nullopt) {
    REQUIRE(tensor.setAttribute("sampleAxis", sampleAxis) == Result::SUCCESS);
    if (batchAxis) {
        REQUIRE(tensor.setAttribute("batchAxis", *batchAxis) == Result::SUCCESS);
    }
    if (channelAxis) {
        REQUIRE(tensor.setAttribute("channelAxis", *channelAxis) == Result::SUCCESS);
    }
}

void SetRealValues(Tensor tensor, const std::vector<F32>& values) {
    REQUIRE(tensor.size() == values.size());
    REQUIRE((tensor.dtype() == DataType::F32 || tensor.dtype() == DataType::CF32));
    for (U64 i = 0; i < values.size(); ++i) {
        if (tensor.dtype() == DataType::F32) {
            tensor.at<F32>(i) = values[i];
        } else {
            tensor.at<CF32>(i) = CF32{values[i], 0.0f};
        }
    }
}

void RequireRealValues(const Tensor& tensor, const std::vector<F32>& expected) {
    REQUIRE(tensor.dtype() == DataType::CF32);
    REQUIRE(tensor.size() == expected.size());
    for (U64 i = 0; i < expected.size(); ++i) {
        REQUIRE_THAT(tensor.at<CF32>(i).real(),
                     Catch::Matchers::WithinAbs(expected[i], 1e-5f));
        REQUIRE_THAT(tensor.at<CF32>(i).imag(),
                     Catch::Matchers::WithinAbs(0.0f, 1e-5f));
    }
}

std::pair<std::string, std::string> CreateRealSource(Flowgraph& flowgraph,
                                                     const std::string& name,
                                                     const std::string& dataType,
                                                     const std::vector<F32>& values) {
    REQUIRE((dataType == "F32" || dataType == "CF32"));

    TestFlowgraph::SyntheticSourceBlockConfig sourceConfig;
    sourceConfig.bufferSize = values.size();
    sourceConfig.value = 0.0f;
    const std::string sourceName = name + "_source";
    REQUIRE(flowgraph.blockCreate(sourceName, sourceConfig, {}) == Result::SUCCESS);
    Tensor source = ViewBlock(flowgraph, sourceName).outputs.at("signal").tensor;
    TagSignal(source, 0);
    SetRealValues(source, values);

    Blocks::Cast castConfig;
    castConfig.outputType = dataType;
    TensorMap castInputs;
    castInputs["buffer"].requested(sourceName, "signal");
    REQUIRE(flowgraph.blockCreate(name, castConfig, castInputs) == Result::SUCCESS);
    return {name, "buffer"};
}

}  // namespace

TEST_CASE_METHOD(FlowgraphFixture, "Filter engine chain", "[modules][dsp][filter_engine]") {
    Blocks::OnesTensor signalSource;
    signalSource.shape = {512};
    signalSource.dataType = "F32";
    REQUIRE(flowgraph->blockCreate("taps_signal", signalSource, {}) == Result::SUCCESS);
    TagSignal(viewBlock("taps_signal").outputs.at("buffer").tensor, 0);
    REQUIRE(flowgraph->blockCreate("taps_filter", "filter_taps", {}, {}) == Result::SUCCESS);
    TagSignal(viewBlock("taps_filter").outputs.at("coeffs").tensor, 1,
              std::nullopt, 0);

    TensorMap engineInputs;
    engineInputs["signal"].requested("taps_signal", "buffer");
    engineInputs["filter"].requested("taps_filter", "coeffs");
    REQUIRE(flowgraph->blockCreate("engine1", "filter_engine", {}, engineInputs) == Result::SUCCESS);
    REQUIRE(viewBlock("engine1").state == Block::State::Created);

    SECTION("disconnecting filter input marks engine incomplete") {
        auto result = flowgraph->blockDisconnect("engine1", "filter");
        REQUIRE((result == Result::SUCCESS || result == Result::INCOMPLETE));
        REQUIRE(viewBlock("engine1").state == Block::State::Incomplete);
        REQUIRE(viewBlock("engine1").outputs.empty());
    }

    SECTION("reconnecting filter input restores engine") {
        flowgraph->blockDisconnect("engine1", "filter");
        REQUIRE(viewBlock("engine1").state == Block::State::Incomplete);

        auto result = flowgraph->blockConnect("engine1", "filter", "taps_filter", "coeffs");
        REQUIRE(result == Result::SUCCESS);
        REQUIRE(viewBlock("engine1").state == Block::State::Created);
    }
}

TEST_CASE_METHOD(FlowgraphFixture,
                 "Filter engine remains incomplete without bound inputs",
                 "[modules][dsp][filter_engine][lifecycle]") {
    Blocks::FilterEngine config;

    SECTION("missing inputs") {
        REQUIRE(flowgraph->blockCreate("missing_inputs", config, {}) == Result::SUCCESS);

        const auto block = viewBlock("missing_inputs");
        REQUIRE(block.state == Block::State::Incomplete);
        REQUIRE(block.interfaceInputs.size() == 2);
        REQUIRE(block.interfaceOutputs.size() == 1);
        REQUIRE(block.outputs.empty());
    }

    SECTION("unresolved filter input") {
        Blocks::OnesTensor source;
        source.shape = {8};
        source.dataType = "CF32";
        REQUIRE(flowgraph->blockCreate("unresolved_src", source, {}) == Result::SUCCESS);
        TagSignal(viewBlock("unresolved_src").outputs.at("buffer").tensor, 0);

        TensorMap inputs;
        inputs["signal"].requested("unresolved_src", "buffer");
        inputs["filter"].requested("unresolved_src", "missing");
        REQUIRE(flowgraph->blockCreate("unresolved_filter", config, inputs) ==
                Result::SUCCESS);

        const auto block = viewBlock("unresolved_filter");
        REQUIRE(block.state == Block::State::Incomplete);
        REQUIRE(block.interfaceInputs.size() == 2);
        REQUIRE(block.interfaceOutputs.size() == 1);
        REQUIRE(block.outputs.empty());
    }
}

TEST_CASE_METHOD(FlowgraphFixture,
                 "Filter engine rejects invalid bound rank and extent before define",
                 "[modules][dsp][filter_engine][validation]") {
    Blocks::OnesTensor signalSource;
    signalSource.shape = {8};
    signalSource.dataType = "CF32";
    REQUIRE(flowgraph->blockCreate("geometry_signal", signalSource, {}) ==
            Result::SUCCESS);
    TagSignal(viewBlock("geometry_signal").outputs.at("buffer").tensor, 0);

    Blocks::OnesTensor filterSource;
    filterSource.shape = {3};
    filterSource.dataType = "CF32";
    REQUIRE(flowgraph->blockCreate("geometry_filter", filterSource, {}) ==
            Result::SUCCESS);
    TagSignal(viewBlock("geometry_filter").outputs.at("buffer").tensor, 0);

    Tensor malformed;
    SECTION("rank-zero signal") {
        malformed = viewBlock("geometry_signal").outputs.at("buffer").tensor;
        REQUIRE(malformed.slice(std::vector<Token>{Token{U64{0}}}) ==
                Result::SUCCESS);
        REQUIRE(malformed.rank() == 0);
    }
    SECTION("rank-zero filter") {
        malformed = viewBlock("geometry_filter").outputs.at("buffer").tensor;
        REQUIRE(malformed.slice(std::vector<Token>{Token{U64{0}}}) ==
                Result::SUCCESS);
        REQUIRE(malformed.rank() == 0);
    }
    SECTION("zero-extent signal") {
        malformed = viewBlock("geometry_signal").outputs.at("buffer").tensor;
        REQUIRE(malformed.slice(std::vector<Token>{
                    Token{U64{0}, U64{0}, U64{1}, true}}) == Result::SUCCESS);
        REQUIRE(malformed.shape() == Shape{0});
    }
    SECTION("zero-extent filter") {
        malformed = viewBlock("geometry_filter").outputs.at("buffer").tensor;
        REQUIRE(malformed.slice(std::vector<Token>{
                    Token{U64{0}, U64{0}, U64{1}, true}}) == Result::SUCCESS);
        REQUIRE(malformed.shape() == Shape{0});
    }
    SECTION("combined extent overflow") {
        malformed = viewBlock("geometry_signal").outputs.at("buffer").tensor;
        REQUIRE(malformed.slice(std::vector<Token>{
                    Token{U64{0}, U64{1}, U64{1}, true}}) == Result::SUCCESS);
        REQUIRE(malformed.broadcastTo(
                    Shape{0, std::numeric_limits<U64>::max()}) == Result::SUCCESS);
        TagSignal(malformed, 1, 0);
    }

    TensorMap inputs;
    inputs["signal"].requested("geometry_signal", "buffer");
    inputs["filter"].requested("geometry_filter", "buffer");
    REQUIRE(flowgraph->blockCreate("geometry_bad", Blocks::FilterEngine{}, inputs) ==
            Result::SUCCESS);
    RequireErroredWithInterface(viewBlock("geometry_bad"));
}

TEST_CASE_METHOD(FlowgraphFixture,
                  "Filter engine rejects malformed filter coefficient roles",
                  "[modules][dsp][filter_engine][validation]") {
    Blocks::OnesTensor signalSource;
    signalSource.shape = {8};
    signalSource.dataType = "CF32";
    REQUIRE(flowgraph->blockCreate("rank_filter_signal", signalSource, {}) ==
            Result::SUCCESS);
    TagSignal(viewBlock("rank_filter_signal").outputs.at("buffer").tensor, 0);

    Blocks::OnesTensor filterSource;
    filterSource.shape = {2, 3};
    filterSource.dataType = "CF32";
    SECTION("rank-three filter") {
        filterSource.shape = {1, 2, 3};
        REQUIRE(flowgraph->blockCreate("rank_filter_coeffs", filterSource, {}) ==
                Result::SUCCESS);
        TagSignal(viewBlock("rank_filter_coeffs").outputs.at("buffer").tensor,
                  2, std::nullopt, 1);
    }
    SECTION("untagged rank-two filter") {
        REQUIRE(flowgraph->blockCreate("rank_filter_coeffs", filterSource, {}) ==
                Result::SUCCESS);
    }
    SECTION("reversed rank-two roles") {
        REQUIRE(flowgraph->blockCreate("rank_filter_coeffs", filterSource, {}) ==
                Result::SUCCESS);
        TagSignal(viewBlock("rank_filter_coeffs").outputs.at("buffer").tensor,
                  0, std::nullopt, 1);
    }
    SECTION("batched filter") {
        REQUIRE(flowgraph->blockCreate("rank_filter_coeffs", filterSource, {}) ==
                Result::SUCCESS);
        Tensor filter = viewBlock("rank_filter_coeffs").outputs.at("buffer").tensor;
        TagSignal(filter, 1, std::nullopt, 0);
        REQUIRE(filter.setAttribute("batchAxis", Index{0}) == Result::SUCCESS);
    }

    TensorMap inputs;
    inputs["signal"].requested("rank_filter_signal", "buffer");
    inputs["filter"].requested("rank_filter_coeffs", "buffer");
    REQUIRE(flowgraph->blockCreate("rank_filter_bad",
                                   Blocks::FilterEngine{},
                                   inputs) == Result::SUCCESS);
    RequireErroredWithInterface(viewBlock("rank_filter_bad"));
}

TEST_CASE_METHOD(FlowgraphFixture,
                  "Filter engine rejects invalid signal metadata",
                  "[modules][dsp][filter_engine][validation]") {
    Blocks::OnesTensor signalSource;
    signalSource.shape = {8, 2};
    signalSource.dataType = "CF32";

    SECTION("batch axis has the wrong type") {
        REQUIRE(flowgraph->blockCreate("layout_signal", signalSource, {}) ==
                Result::SUCCESS);
        Tensor tensor = viewBlock("layout_signal").outputs.at("buffer").tensor;
        TagSignal(tensor, 0);
        REQUIRE(tensor.setAttribute("batchAxis", I64{1}) == Result::SUCCESS);
    }
    SECTION("batch axis is out of bounds") {
        REQUIRE(flowgraph->blockCreate("layout_signal", signalSource, {}) ==
                Result::SUCCESS);
        Tensor tensor = viewBlock("layout_signal").outputs.at("buffer").tensor;
        TagSignal(tensor, 0);
        REQUIRE(tensor.setAttribute("batchAxis", Index{2}) == Result::SUCCESS);
    }
    SECTION("sample axis is missing") {
        REQUIRE(flowgraph->blockCreate("layout_signal", signalSource, {}) ==
                Result::SUCCESS);
    }
    SECTION("sample axis has the wrong type") {
        REQUIRE(flowgraph->blockCreate("layout_signal", signalSource, {}) ==
                Result::SUCCESS);
        Tensor tensor = viewBlock("layout_signal").outputs.at("buffer").tensor;
        REQUIRE(tensor.setAttribute("sampleAxis", I64{0}) == Result::SUCCESS);
    }
    SECTION("sample and batch roles conflict") {
        REQUIRE(flowgraph->blockCreate("layout_signal", signalSource, {}) ==
                Result::SUCCESS);
        Tensor tensor = viewBlock("layout_signal").outputs.at("buffer").tensor;
        TagSignal(tensor, 0);
        REQUIRE(tensor.setAttribute("batchAxis", Index{0}) == Result::SUCCESS);
    }
    Blocks::OnesTensor filterSource;
    filterSource.shape = {3};
    filterSource.dataType = "CF32";
    REQUIRE(flowgraph->blockCreate("layout_filter", filterSource, {}) ==
            Result::SUCCESS);
    TagSignal(viewBlock("layout_filter").outputs.at("buffer").tensor, 0);

    TensorMap inputs;
    inputs["signal"].requested("layout_signal", "buffer");
    inputs["filter"].requested("layout_filter", "buffer");
    REQUIRE(flowgraph->blockCreate("layout_bad", Blocks::FilterEngine{}, inputs) ==
            Result::SUCCESS);
    RequireErroredWithInterface(viewBlock("layout_bad"));
}

TEST_CASE_METHOD(FlowgraphFixture,
                  "Filter engine supports leading and trailing batches",
                  "[modules][dsp][filter_engine][metadata]") {
    Blocks::OnesTensor signalSource;
    signalSource.dataType = "CF32";

    Index batchAxis = 0;
    SECTION("leading batch axis") {
        signalSource.shape = {3, 4};
        batchAxis = 0;
    }
    SECTION("trailing batch axis") {
        signalSource.shape = {4, 3};
        batchAxis = 1;
    }

    REQUIRE(flowgraph->blockCreate("batch_signal", signalSource, {}) ==
            Result::SUCCESS);
    Tensor signal = viewBlock("batch_signal").outputs.at("buffer").tensor;
    const Index sampleAxis = batchAxis == 0 ? Index{1} : Index{0};
    TagSignal(signal, sampleAxis, batchAxis);

    Blocks::OnesTensor filterSource;
    filterSource.shape = {3};
    filterSource.dataType = "CF32";
    REQUIRE(flowgraph->blockCreate("batch_filter", filterSource, {}) ==
            Result::SUCCESS);
    TagSignal(viewBlock("batch_filter").outputs.at("buffer").tensor, 0);

    TensorMap inputs;
    inputs["signal"].requested("batch_signal", "buffer");
    inputs["filter"].requested("batch_filter", "buffer");
    REQUIRE(flowgraph->blockCreate("batch_engine", Blocks::FilterEngine{}, inputs) ==
            Result::SUCCESS);
    REQUIRE(flowgraph->compute() == Result::SUCCESS);

    const Tensor output = viewBlock("batch_engine").outputs.at("buffer").tensor;
    REQUIRE(output.shape() == signalSource.shape);
    REQUIRE(output.attribute("batchAxis").type() == typeid(Index));
    REQUIRE(std::any_cast<Index>(output.attribute("batchAxis")) == batchAxis);
    REQUIRE(std::any_cast<Index>(output.attribute("sampleAxis")) == sampleAxis);
    REQUIRE_FALSE(output.hasAttribute("channelAxis"));
}

TEST_CASE_METHOD(FlowgraphFixture,
                  "Filter engine supports unbatched and batched multi-head layouts",
                  "[modules][dsp][filter_engine][metadata]") {
    Blocks::OnesTensor signalSource;
    signalSource.dataType = "CF32";

    bool batched = false;
    Index batchAxis = 0;
    Index expectedBatchAxis = 0;
    Shape expectedShape;
    SECTION("unbatched") {
        signalSource.shape = {4};
        expectedShape = {2, 4};
    }
    SECTION("leading batch axis") {
        signalSource.shape = {3, 4};
        batched = true;
        batchAxis = 0;
        expectedBatchAxis = 0;
        expectedShape = {3, 2, 4};
    }
    SECTION("trailing batch axis") {
        signalSource.shape = {4, 3};
        batched = true;
        batchAxis = 1;
        expectedBatchAxis = 2;
        expectedShape = {2, 4, 3};
    }

    REQUIRE(flowgraph->blockCreate("multi_signal", signalSource, {}) ==
            Result::SUCCESS);
    Tensor signal = viewBlock("multi_signal").outputs.at("buffer").tensor;
    const Index signalSampleAxis = batched && batchAxis == 0 ? Index{1} : Index{0};
    TagSignal(signal, signalSampleAxis,
              batched ? std::optional<Index>{batchAxis} : std::nullopt);

    Blocks::OnesTensor filterSource;
    filterSource.shape = {2, 3};
    filterSource.dataType = "CF32";
    REQUIRE(flowgraph->blockCreate("multi_filter", filterSource, {}) ==
            Result::SUCCESS);
    TagSignal(viewBlock("multi_filter").outputs.at("buffer").tensor,
              1, std::nullopt, 0);

    TensorMap inputs;
    inputs["signal"].requested("multi_signal", "buffer");
    inputs["filter"].requested("multi_filter", "buffer");
    REQUIRE(flowgraph->blockCreate("multi_engine", Blocks::FilterEngine{}, inputs) ==
            Result::SUCCESS);
    REQUIRE(flowgraph->compute() == Result::SUCCESS);

    const Tensor output = viewBlock("multi_engine").outputs.at("buffer").tensor;
    REQUIRE(output.shape() == expectedShape);
    REQUIRE(std::any_cast<Index>(output.attribute("channelAxis")) ==
            (batched && batchAxis == 0 ? Index{1} : Index{0}));
    REQUIRE(std::any_cast<Index>(output.attribute("sampleAxis")) ==
            (batched && batchAxis == 0 ? Index{2} : Index{1}));
    if (batched) {
        REQUIRE(output.attribute("batchAxis").type() == typeid(Index));
        REQUIRE(std::any_cast<Index>(output.attribute("batchAxis")) ==
                expectedBatchAxis);
    } else {
        REQUIRE_FALSE(output.hasAttribute("batchAxis"));
    }
}

TEST_CASE_METHOD(FlowgraphFixture,
                  "Filter engine preserves signal channels with rank-1 taps",
                  "[modules][dsp][filter_engine][metadata]") {
    Blocks::OnesTensor signalSource;
    signalSource.shape = {3, 2, 8};
    signalSource.dataType = "CF32";
    REQUIRE(flowgraph->blockCreate("channel_signal", signalSource, {}) ==
            Result::SUCCESS);
    TagSignal(viewBlock("channel_signal").outputs.at("buffer").tensor,
              2, 0, 1);

    Blocks::OnesTensor filterSource;
    filterSource.shape = {3};
    filterSource.dataType = "CF32";
    REQUIRE(flowgraph->blockCreate("channel_filter", filterSource, {}) ==
            Result::SUCCESS);
    TagSignal(viewBlock("channel_filter").outputs.at("buffer").tensor, 0);

    TensorMap inputs;
    inputs["signal"].requested("channel_signal", "buffer");
    inputs["filter"].requested("channel_filter", "buffer");
    REQUIRE(flowgraph->blockCreate("channel_engine", Blocks::FilterEngine{}, inputs) ==
            Result::SUCCESS);
    REQUIRE(flowgraph->compute() == Result::SUCCESS);

    const Tensor output = viewBlock("channel_engine").outputs.at("buffer").tensor;
    REQUIRE(output.shape() == Shape{3, 2, 8});
    REQUIRE(std::any_cast<Index>(output.attribute("sampleAxis")) == 2);
    REQUIRE(std::any_cast<Index>(output.attribute("batchAxis")) == 0);
    REQUIRE(std::any_cast<Index>(output.attribute("channelAxis")) == 1);
}

TEST_CASE_METHOD(FlowgraphFixture,
                  "Filter engine rejects nested filter channels",
                  "[modules][dsp][filter_engine][metadata][validation]") {
    Blocks::OnesTensor signalSource;
    signalSource.shape = {2, 8};
    signalSource.dataType = "CF32";
    REQUIRE(flowgraph->blockCreate("nested_signal", signalSource, {}) ==
            Result::SUCCESS);
    TagSignal(viewBlock("nested_signal").outputs.at("buffer").tensor,
              1, std::nullopt, 0);

    Blocks::OnesTensor filterSource;
    filterSource.shape = {3, 3};
    filterSource.dataType = "CF32";
    REQUIRE(flowgraph->blockCreate("nested_filter", filterSource, {}) ==
            Result::SUCCESS);
    TagSignal(viewBlock("nested_filter").outputs.at("buffer").tensor,
              1, std::nullopt, 0);

    TensorMap inputs;
    inputs["signal"].requested("nested_signal", "buffer");
    inputs["filter"].requested("nested_filter", "buffer");
    REQUIRE(flowgraph->blockCreate("nested_engine", Blocks::FilterEngine{}, inputs) ==
            Result::SUCCESS);
    RequireErroredWithInterface(viewBlock("nested_engine"));
}

TEST_CASE_METHOD(FlowgraphFixture,
                  "Filter engine rejects malformed optional metadata before define",
                 "[modules][dsp][filter_engine][metadata][validation]") {
    Blocks::OnesTensor signalSource;
    signalSource.shape = {8};
    signalSource.dataType = "CF32";
    REQUIRE(flowgraph->blockCreate("metadata_signal", signalSource, {}) ==
            Result::SUCCESS);
    TagSignal(viewBlock("metadata_signal").outputs.at("buffer").tensor, 0);

    Blocks::OnesTensor filterSource;
    filterSource.shape = {3};
    filterSource.dataType = "CF32";
    REQUIRE(flowgraph->blockCreate("metadata_filter", filterSource, {}) ==
            Result::SUCCESS);
    TagSignal(viewBlock("metadata_filter").outputs.at("buffer").tensor, 0);

    std::any sampleRate = F32{8.0f};
    std::any bandwidth = F32{4.0f};
    std::any center = F32{0.0f};
    SECTION("sample rate is not F32") {
        sampleRate = F64{8.0};
    }
    SECTION("bandwidth is not F32") {
        bandwidth = F64{4.0};
    }
    SECTION("center is not F32") {
        center = F64{0.0};
    }

    Tensor filter = viewBlock("metadata_filter").outputs.at("buffer").tensor;
    REQUIRE(filter.setAttribute("sampleRate", sampleRate) == Result::SUCCESS);
    REQUIRE(filter.setAttribute("bandwidth", bandwidth) == Result::SUCCESS);
    REQUIRE(filter.setAttribute("center", center) == Result::SUCCESS);

    TensorMap inputs;
    inputs["signal"].requested("metadata_signal", "buffer");
    inputs["filter"].requested("metadata_filter", "buffer");
    REQUIRE(flowgraph->blockCreate("metadata_bad", Blocks::FilterEngine{}, inputs) ==
            Result::SUCCESS);
    RequireErroredWithInterface(viewBlock("metadata_bad"));
}

TEST_CASE_METHOD(FlowgraphFixture,
                 "Filter engine rejects unsafe center index conversion before define",
                 "[modules][dsp][filter_engine][metadata][validation]") {
    Blocks::OnesTensor signalSource;
    signalSource.shape = {8};
    signalSource.dataType = "CF32";
    REQUIRE(flowgraph->blockCreate("center_signal", signalSource, {}) ==
            Result::SUCCESS);
    TagSignal(viewBlock("center_signal").outputs.at("buffer").tensor, 0);

    Blocks::OnesTensor filterSource;
    filterSource.shape = {3};
    filterSource.dataType = "CF32";
    REQUIRE(flowgraph->blockCreate("center_filter", filterSource, {}) ==
            Result::SUCCESS);
    TagSignal(viewBlock("center_filter").outputs.at("buffer").tensor, 0);

    F32 center = 0.0f;
    SECTION("non-finite fold index") {
        center = std::numeric_limits<F32>::infinity();
    }
    SECTION("fold index outside U64 range") {
        center = 1.0e20f;
    }

    Tensor filter = viewBlock("center_filter").outputs.at("buffer").tensor;
    REQUIRE(filter.setAttribute("sampleRate", F32{8.0f}) == Result::SUCCESS);
    REQUIRE(filter.setAttribute("bandwidth", F32{4.0f}) == Result::SUCCESS);
    REQUIRE(filter.setAttribute("center", center) == Result::SUCCESS);

    TensorMap inputs;
    inputs["signal"].requested("center_signal", "buffer");
    inputs["filter"].requested("center_filter", "buffer");
    REQUIRE(flowgraph->blockCreate("center_bad", Blocks::FilterEngine{}, inputs) ==
            Result::SUCCESS);
    RequireErroredWithInterface(viewBlock("center_bad"));
}

TEST_CASE_METHOD(FlowgraphFixture,
                 "Filter engine translates rounded center bins with negated fold offsets",
                 "[modules][dsp][filter_engine][numeric][resample]") {
    const auto signalOutput = CreateRealSource(
        *flowgraph,
        "center_signal",
        "CF32",
        {1.0f, -0.5f, -0.5f, 1.0f});
    const auto filterOutput = CreateRealSource(
        *flowgraph, "center_taps", "CF32", {1.0f, 0.0f, 0.0f});

    F32 center = 0.0f;
    F32 imaginarySign = 0.0f;
    SECTION("positive center") {
        center = 1.6f;
        imaginarySign = -1.0f;
    }
    SECTION("negative center") {
        center = -1.6f;
        imaginarySign = 1.0f;
    }
    SECTION("wrapped negative center") {
        center = -7.0f;
        imaginarySign = -1.0f;
    }

    Tensor filter = viewBlock(filterOutput.first).outputs.at(filterOutput.second).tensor;
    REQUIRE(filter.setAttribute("sampleRate", F32{6.0f}) == Result::SUCCESS);
    REQUIRE(filter.setAttribute("bandwidth", F32{3.0f}) == Result::SUCCESS);
    REQUIRE(filter.setAttribute("center", center) == Result::SUCCESS);

    TensorMap inputs;
    inputs["signal"].requested(signalOutput.first, signalOutput.second);
    inputs["filter"].requested(filterOutput.first, filterOutput.second);
    REQUIRE(flowgraph->blockCreate("center_engine",
                                   Blocks::FilterEngine{},
                                   inputs) == Result::SUCCESS);
    REQUIRE(flowgraph->compute() == Result::SUCCESS);

    const Tensor output = viewBlock("center_engine").outputs.at("buffer").tensor;
    REQUIRE(output.shape() == Shape{2});
    REQUIRE_THAT(output.at<CF32>(0).real(),
                 Catch::Matchers::WithinAbs(1.0f, 1e-5f));
    REQUIRE_THAT(output.at<CF32>(1).real(),
                 Catch::Matchers::WithinAbs(0.25f, 1e-5f));
    REQUIRE_THAT(output.at<CF32>(1).imag(),
                 Catch::Matchers::WithinAbs(imaginarySign * 0.4330127f, 1e-5f));
}

TEST_CASE_METHOD(FlowgraphFixture,
                 "Filter engine safely bypasses an unrepresentable resampler ratio",
                 "[modules][dsp][filter_engine][metadata][validation]") {
    Blocks::OnesTensor signalSource;
    signalSource.shape = {8};
    signalSource.dataType = "CF32";
    REQUIRE(flowgraph->blockCreate("ratio_signal", signalSource, {}) ==
            Result::SUCCESS);
    TagSignal(viewBlock("ratio_signal").outputs.at("buffer").tensor, 0);

    Blocks::OnesTensor filterSource;
    filterSource.shape = {3};
    filterSource.dataType = "CF32";
    REQUIRE(flowgraph->blockCreate("ratio_filter", filterSource, {}) ==
            Result::SUCCESS);
    TagSignal(viewBlock("ratio_filter").outputs.at("buffer").tensor, 0);

    Tensor filter = viewBlock("ratio_filter").outputs.at("buffer").tensor;
    REQUIRE(filter.setAttribute("sampleRate", F32{1.0e20f}) == Result::SUCCESS);
    REQUIRE(filter.setAttribute("bandwidth", F32{1.0f}) == Result::SUCCESS);
    REQUIRE(filter.setAttribute("center", F32{0.0f}) == Result::SUCCESS);

    TensorMap inputs;
    inputs["signal"].requested("ratio_signal", "buffer");
    inputs["filter"].requested("ratio_filter", "buffer");
    REQUIRE(flowgraph->blockCreate("ratio_engine", Blocks::FilterEngine{}, inputs) ==
            Result::SUCCESS);

    const auto block = viewBlock("ratio_engine");
    REQUIRE(block.state == Block::State::Created);
    REQUIRE(block.outputs.at("buffer").tensor.shape() == Shape{8});
    REQUIRE(flowgraph->compute() == Result::SUCCESS);
}

TEST_CASE_METHOD(FlowgraphFixture,
                 "Filter engine recreates to recover from an unsafe reconnection",
                 "[modules][dsp][filter_engine][connection][validation]") {
    Blocks::OnesTensor signalSource;
    signalSource.shape = {8};
    signalSource.dataType = "CF32";
    REQUIRE(flowgraph->blockCreate("rollback_signal", signalSource, {}) ==
            Result::SUCCESS);
    TagSignal(viewBlock("rollback_signal").outputs.at("buffer").tensor, 0);

    Blocks::OnesTensor filterSource;
    filterSource.shape = {3};
    filterSource.dataType = "CF32";
    REQUIRE(flowgraph->blockCreate("rollback_filter", filterSource, {}) ==
            Result::SUCCESS);
    REQUIRE(flowgraph->blockCreate("rollback_bad_filter", filterSource, {}) ==
            Result::SUCCESS);
    TagSignal(viewBlock("rollback_filter").outputs.at("buffer").tensor, 0);
    TagSignal(viewBlock("rollback_bad_filter").outputs.at("buffer").tensor, 0);

    Tensor validFilter = viewBlock("rollback_filter").outputs.at("buffer").tensor;
    REQUIRE(validFilter.setAttribute("sampleRate", F32{8.0f}) == Result::SUCCESS);
    REQUIRE(validFilter.setAttribute("bandwidth", F32{4.0f}) == Result::SUCCESS);
    REQUIRE(validFilter.setAttribute("center", F32{0.0f}) == Result::SUCCESS);

    Tensor invalidFilter =
        viewBlock("rollback_bad_filter").outputs.at("buffer").tensor;
    REQUIRE(invalidFilter.setAttribute("sampleRate", F32{8.0f}) == Result::SUCCESS);
    REQUIRE(invalidFilter.setAttribute("bandwidth", F32{4.0f}) == Result::SUCCESS);
    REQUIRE(invalidFilter.setAttribute("center", F32{1.0e20f}) == Result::SUCCESS);

    TensorMap inputs;
    inputs["signal"].requested("rollback_signal", "buffer");
    inputs["filter"].requested("rollback_filter", "buffer");
    REQUIRE(flowgraph->blockCreate("rollback_engine", Blocks::FilterEngine{}, inputs) ==
            Result::SUCCESS);
    REQUIRE(flowgraph->compute() == Result::SUCCESS);

    const Index outputId =
        viewBlock("rollback_engine").outputs.at("buffer").tensor.id();
    REQUIRE(flowgraph->blockConnect("rollback_engine",
                                    "filter",
                                    "rollback_bad_filter",
                                    "buffer") == Result::SUCCESS);

    const auto rejected = viewBlock("rollback_engine");
    REQUIRE(rejected.state == Block::State::Errored);
    REQUIRE(rejected.outputs.empty());
    REQUIRE(rejected.inputs.at("filter").external.has_value());
    REQUIRE(rejected.inputs.at("filter").external->block == "rollback_bad_filter");

    REQUIRE(flowgraph->blockConnect("rollback_engine",
                                    "filter",
                                    "rollback_filter",
                                    "buffer") == Result::SUCCESS);

    const auto block = viewBlock("rollback_engine");
    REQUIRE(block.state == Block::State::Created);
    REQUIRE(block.outputs.at("buffer").tensor.id() != outputId);
    REQUIRE(block.inputs.at("filter").external.has_value());
    REQUIRE(block.inputs.at("filter").external->block == "rollback_filter");

    Tensor output = block.outputs.at("buffer").tensor;
    const CF32 sentinel{-123.0f, 45.0f};
    std::fill(output.data<CF32>(), output.data<CF32>() + output.size(), sentinel);
    REQUIRE(flowgraph->compute() == Result::SUCCESS);
    REQUIRE(std::any_of(output.data<CF32>(),
                        output.data<CF32>() + output.size(),
                        [&](const CF32 value) { return value != sentinel; }));
}

TEST_CASE_METHOD(FlowgraphFixture,
                   "Filter engine uses full complex spectra for every operand type",
                   "[modules][dsp][filter_engine][numeric][F32]") {
    const char* signalDataType = nullptr;
    const char* filterDataType = nullptr;
    std::vector<F32> signalValues = {1.0f, 2.0f, 3.0f};
    std::vector<F32> filterValues;
    std::vector<F32> firstExpected;
    std::vector<F32> secondExpected;
    SECTION("F32 signal and F32 filter") {
        signalDataType = "F32";
        filterDataType = "F32";
        filterValues = {4.0f, 5.0f};
        firstExpected = {4.0f, 13.0f, 22.0f};
        secondExpected = {19.0f, 13.0f, 22.0f};
    }
    SECTION("F32 signal and CF32 filter") {
        signalDataType = "F32";
        filterDataType = "CF32";
        filterValues = {4.0f, 5.0f, 6.0f};
        firstExpected = {4.0f, 13.0f, 28.0f};
        secondExpected = {31.0f, 31.0f, 28.0f};
    }
    SECTION("CF32 signal and F32 filter") {
        signalDataType = "CF32";
        filterDataType = "F32";
        filterValues = {4.0f, 5.0f};
        firstExpected = {4.0f, 13.0f, 22.0f};
        secondExpected = {19.0f, 13.0f, 22.0f};
    }
    SECTION("CF32 signal and CF32 filter") {
        signalDataType = "CF32";
        filterDataType = "CF32";
        filterValues = {4.0f, 5.0f, 6.0f};
        firstExpected = {4.0f, 13.0f, 28.0f};
        secondExpected = {31.0f, 31.0f, 28.0f};
    }

    const auto signalOutput = CreateRealSource(
        *flowgraph, "normalize_signal", signalDataType, signalValues);
    const auto filterOutput = CreateRealSource(
        *flowgraph, "normalize_taps", filterDataType, filterValues);

    TensorMap inputs;
    inputs["signal"].requested(signalOutput.first, signalOutput.second);
    inputs["filter"].requested(filterOutput.first, filterOutput.second);
    REQUIRE(flowgraph->blockCreate("normalize_engine",
                                   Blocks::FilterEngine{},
                                   inputs) == Result::SUCCESS);
    REQUIRE(flowgraph->compute() == Result::SUCCESS);

    const Tensor output =
        viewBlock("normalize_engine").outputs.at("buffer").tensor;
    REQUIRE(output.shape() == Shape{3});
    REQUIRE(std::any_cast<Index>(output.attribute("sampleAxis")) == 0);
    REQUIRE_FALSE(output.hasAttribute("batchAxis"));
    REQUIRE_FALSE(output.hasAttribute("channelAxis"));
    RequireRealValues(output, firstExpected);

    REQUIRE(flowgraph->compute() == Result::SUCCESS);
    RequireRealValues(output, secondExpected);
}

TEST_CASE_METHOD(FlowgraphFixture,
                 "Filter engine supports one tap without overlap-add",
                 "[modules][dsp][filter_engine][numeric]") {
    const auto signalOutput = CreateRealSource(
        *flowgraph, "one_tap_signal", "CF32", {1.0f, 2.0f, 3.0f, 4.0f});
    const auto filterOutput = CreateRealSource(
        *flowgraph, "one_tap_filter", "CF32", {1.0f});
    Tensor filter = viewBlock(filterOutput.first).outputs.at(filterOutput.second).tensor;
    REQUIRE(filter.setAttribute("sampleRate", F32{8.0f}) == Result::SUCCESS);
    REQUIRE(filter.setAttribute("bandwidth", F32{4.0f}) == Result::SUCCESS);
    REQUIRE(filter.setAttribute("center", F32{0.0f}) == Result::SUCCESS);

    TensorMap inputs;
    inputs["signal"].requested(signalOutput.first, signalOutput.second);
    inputs["filter"].requested(filterOutput.first, filterOutput.second);
    REQUIRE(flowgraph->blockCreate("one_tap_engine",
                                   Blocks::FilterEngine{},
                                   inputs) == Result::SUCCESS);
    REQUIRE(viewBlock("one_tap_engine").state == Block::State::Created);
    REQUIRE(flowgraph->compute() == Result::SUCCESS);

    const Tensor output =
        viewBlock("one_tap_engine").outputs.at("buffer").tensor;
    REQUIRE(output.shape() == Shape{2});
    REQUIRE(std::any_cast<Index>(output.attribute("sampleAxis")) == 0);
    REQUIRE_FALSE(output.hasAttribute("batchAxis"));
    REQUIRE_FALSE(output.hasAttribute("channelAxis"));
    REQUIRE(std::any_cast<F32>(output.attribute("sampleRate")) == 4.0f);
    RequireRealValues(output, {1.0f, 3.0f});
}

TEST_CASE_METHOD(FlowgraphFixture,
                   "Filter engine keeps F32 resampling on full complex spectra",
                  "[modules][dsp][filter_engine][numeric][F32][resample]") {
    const auto signalOutput = CreateRealSource(
        *flowgraph, "resample_signal", "F32", {1.0f, 2.0f, 3.0f, 0.0f});
    const auto filterOutput = CreateRealSource(
        *flowgraph, "resample_taps", "F32", {4.0f, 5.0f, 6.0f});
    Tensor filter = viewBlock(filterOutput.first).outputs.at(filterOutput.second).tensor;
    REQUIRE(filter.setAttribute("sampleRate", F32{8.0f}) == Result::SUCCESS);
    REQUIRE(filter.setAttribute("bandwidth", F32{4.0f}) == Result::SUCCESS);
    REQUIRE(filter.setAttribute("center", F32{0.0f}) == Result::SUCCESS);

    TensorMap inputs;
    inputs["signal"].requested(signalOutput.first, signalOutput.second);
    inputs["filter"].requested(filterOutput.first, filterOutput.second);
    REQUIRE(flowgraph->blockCreate("resample_engine",
                                   Blocks::FilterEngine{},
                                   inputs) == Result::SUCCESS);
    REQUIRE(flowgraph->compute() == Result::SUCCESS);

    const Tensor output =
        viewBlock("resample_engine").outputs.at("buffer").tensor;
    REQUIRE(output.shape() == Shape{2});
    REQUIRE(std::any_cast<F32>(output.attribute("sampleRate")) == 4.0f);
    RequireRealValues(output, {4.0f, 28.0f});

    REQUIRE(flowgraph->compute() == Result::SUCCESS);
    RequireRealValues(output, {22.0f, 28.0f});
}
