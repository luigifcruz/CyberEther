#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <any>
#include <limits>
#include <string>
#include <vector>

#include "jetstream/domains/dsp/filter_engine/block.hh"
#include "jetstream/domains/dsp/filter_taps/block.hh"
#include "jetstream/domains/core/ones_tensor/block.hh"
#include "flowgraph_fixture.hh"

using namespace Jetstream;

namespace {

void RequireRejectedBeforeDefine(const Flowgraph::View::BlockData& block) {
    REQUIRE(block.state == Block::State::Errored);
    REQUIRE(block.interfaceInputs.empty());
    REQUIRE(block.interfaceOutputs.empty());
    REQUIRE(block.interfaceConfigs.empty());
    REQUIRE(block.outputs.empty());
}

}  // namespace

TEST_CASE_METHOD(FlowgraphFixture, "Filter engine chain", "[modules][dsp][filter_engine]") {
    REQUIRE(flowgraph->blockCreate("taps_signal", "filter_taps", {}, {}) == Result::SUCCESS);
    REQUIRE(flowgraph->blockCreate("taps_filter", "filter_taps", {}, {}) == Result::SUCCESS);

    TensorMap engineInputs;
    engineInputs["signal"].requested("taps_signal", "coeffs");
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

    Blocks::OnesTensor filterSource;
    filterSource.shape = {3};
    filterSource.dataType = "CF32";
    REQUIRE(flowgraph->blockCreate("geometry_filter", filterSource, {}) ==
            Result::SUCCESS);

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
    }

    TensorMap inputs;
    inputs["signal"].requested("geometry_signal", "buffer");
    inputs["filter"].requested("geometry_filter", "buffer");
    REQUIRE(flowgraph->blockCreate("geometry_bad", Blocks::FilterEngine{}, inputs) ==
            Result::SUCCESS);
    RequireRejectedBeforeDefine(viewBlock("geometry_bad"));
}

TEST_CASE_METHOD(FlowgraphFixture,
                 "Filter engine rejects malformed optional metadata before define",
                 "[modules][dsp][filter_engine][metadata][validation]") {
    Blocks::OnesTensor signalSource;
    signalSource.shape = {8};
    signalSource.dataType = "CF32";
    REQUIRE(flowgraph->blockCreate("metadata_signal", signalSource, {}) ==
            Result::SUCCESS);

    Blocks::OnesTensor filterSource;
    filterSource.shape = {3};
    filterSource.dataType = "CF32";
    REQUIRE(flowgraph->blockCreate("metadata_filter", filterSource, {}) ==
            Result::SUCCESS);

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
    RequireRejectedBeforeDefine(viewBlock("metadata_bad"));
}

TEST_CASE_METHOD(FlowgraphFixture,
                 "Filter engine rejects unsafe center index conversion before define",
                 "[modules][dsp][filter_engine][metadata][validation]") {
    Blocks::OnesTensor signalSource;
    signalSource.shape = {8};
    signalSource.dataType = "CF32";
    REQUIRE(flowgraph->blockCreate("center_signal", signalSource, {}) ==
            Result::SUCCESS);

    Blocks::OnesTensor filterSource;
    filterSource.shape = {3};
    filterSource.dataType = "CF32";
    REQUIRE(flowgraph->blockCreate("center_filter", filterSource, {}) ==
            Result::SUCCESS);

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
    RequireRejectedBeforeDefine(viewBlock("center_bad"));
}

TEST_CASE_METHOD(FlowgraphFixture,
                 "Filter engine negative center wrapping changes the filtered output",
                 "[modules][dsp][filter_engine][metadata]") {
    Blocks::OnesTensor signalSource;
    signalSource.shape = {8};
    signalSource.dataType = "CF32";
    REQUIRE(flowgraph->blockCreate("negative_center_signal", signalSource, {}) ==
            Result::SUCCESS);

    Blocks::OnesTensor filterSource;
    filterSource.shape = {3};
    filterSource.dataType = "CF32";
    REQUIRE(flowgraph->blockCreate("zero_center_filter", filterSource, {}) ==
            Result::SUCCESS);
    REQUIRE(flowgraph->blockCreate("negative_center_filter", filterSource, {}) ==
            Result::SUCCESS);

    Tensor zeroFilter = viewBlock("zero_center_filter").outputs.at("buffer").tensor;
    REQUIRE(zeroFilter.setAttribute("sampleRate", F32{8.0f}) == Result::SUCCESS);
    REQUIRE(zeroFilter.setAttribute("bandwidth", F32{4.0f}) == Result::SUCCESS);
    REQUIRE(zeroFilter.setAttribute("center", F32{0.0f}) == Result::SUCCESS);

    Tensor negativeFilter =
        viewBlock("negative_center_filter").outputs.at("buffer").tensor;
    REQUIRE(negativeFilter.setAttribute("sampleRate", F32{8.0f}) == Result::SUCCESS);
    REQUIRE(negativeFilter.setAttribute("bandwidth", F32{4.0f}) == Result::SUCCESS);
    REQUIRE(negativeFilter.setAttribute("center", F32{-1.6f}) == Result::SUCCESS);

    TensorMap zeroInputs;
    zeroInputs["signal"].requested("negative_center_signal", "buffer");
    zeroInputs["filter"].requested("zero_center_filter", "buffer");
    REQUIRE(flowgraph->blockCreate("zero_center_engine",
                                   Blocks::FilterEngine{},
                                   zeroInputs) == Result::SUCCESS);

    TensorMap negativeInputs;
    negativeInputs["signal"].requested("negative_center_signal", "buffer");
    negativeInputs["filter"].requested("negative_center_filter", "buffer");
    REQUIRE(flowgraph->blockCreate("negative_center_engine",
                                   Blocks::FilterEngine{},
                                   negativeInputs) == Result::SUCCESS);

    REQUIRE(viewBlock("zero_center_engine").state == Block::State::Created);
    REQUIRE(viewBlock("negative_center_engine").state == Block::State::Created);
    REQUIRE(flowgraph->compute() == Result::SUCCESS);

    const Tensor zeroOutput =
        viewBlock("zero_center_engine").outputs.at("buffer").tensor;
    const Tensor negativeOutput =
        viewBlock("negative_center_engine").outputs.at("buffer").tensor;
    REQUIRE(zeroOutput.shape() == Shape{4});
    REQUIRE(negativeOutput.shape() == zeroOutput.shape());
    REQUIRE_FALSE(std::equal(negativeOutput.data<CF32>(),
                             negativeOutput.data<CF32>() + negativeOutput.size(),
                             zeroOutput.data<CF32>()));
}

TEST_CASE_METHOD(FlowgraphFixture,
                 "Filter engine safely bypasses an unrepresentable resampler ratio",
                 "[modules][dsp][filter_engine][metadata][validation]") {
    Blocks::OnesTensor signalSource;
    signalSource.shape = {8};
    signalSource.dataType = "CF32";
    REQUIRE(flowgraph->blockCreate("ratio_signal", signalSource, {}) ==
            Result::SUCCESS);

    Blocks::OnesTensor filterSource;
    filterSource.shape = {3};
    filterSource.dataType = "CF32";
    REQUIRE(flowgraph->blockCreate("ratio_filter", filterSource, {}) ==
            Result::SUCCESS);

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

    Blocks::OnesTensor filterSource;
    filterSource.shape = {3};
    filterSource.dataType = "CF32";
    REQUIRE(flowgraph->blockCreate("rollback_filter", filterSource, {}) ==
            Result::SUCCESS);
    REQUIRE(flowgraph->blockCreate("rollback_bad_filter", filterSource, {}) ==
            Result::SUCCESS);

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
