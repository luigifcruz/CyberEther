#include <catch2/catch_test_macros.hpp>

#include <any>
#include <limits>
#include <string>
#include <vector>

#include "flowgraph_fixture.hh"
#include "jetstream/domains/core/ones_tensor/block.hh"
#include "jetstream/domains/dsp/filter/block.hh"

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

TEST_CASE_METHOD(FlowgraphFixture,
                 "Filter block creates with default config",
                 "[modules][dsp][filter][block]") {
    Parser::Map sourceConfig;
    sourceConfig["signalDataType"] = std::string("CF32");
    sourceConfig["bufferSize"] = std::string("512");

    REQUIRE(flowgraph->blockCreate("src", "signal_generator", sourceConfig, {}) ==
            Result::SUCCESS);

    TensorMap inputs;
    inputs["signal"].requested("src", "signal");

    REQUIRE(flowgraph->blockCreate("filter", "filter", {}, inputs) == Result::SUCCESS);
    REQUIRE(viewBlock("filter").state == Block::State::Created);
    REQUIRE(viewBlock("filter").outputs.contains("buffer"));
}

TEST_CASE_METHOD(FlowgraphFixture,
                 "Filter block rejects zero heads before define",
                 "[modules][dsp][filter][block][validation]") {
    Blocks::Filter config;
    config.heads = 0;

    REQUIRE(flowgraph->blockCreate("filter_zero_heads", config, {}) ==
            Result::SUCCESS);
    RequireRejectedBeforeDefine(viewBlock("filter_zero_heads"));
}

TEST_CASE_METHOD(FlowgraphFixture,
                 "Filter block uses heads over center length",
                 "[modules][dsp][filter][block]") {
    Parser::Map sourceConfig;
    sourceConfig["signalDataType"] = std::string("CF32");
    sourceConfig["bufferSize"] = std::string("512");

    REQUIRE(flowgraph->blockCreate("src", "signal_generator", sourceConfig, {}) ==
            Result::SUCCESS);

    Parser::Map config;
    config["heads"] = std::string("1");
    config["center"] = std::string("[0, 400000]");

    TensorMap inputs;
    inputs["signal"].requested("src", "signal");

    REQUIRE(flowgraph->blockCreate("filter_single", "filter", config, inputs) ==
            Result::SUCCESS);
    REQUIRE(viewBlock("filter_single").state == Block::State::Created);
    REQUIRE(viewBlock("filter_single").outputs.contains("buffer"));
}

TEST_CASE_METHOD(FlowgraphFixture,
                  "Filter block heads can shrink stale center vector",
                  "[modules][dsp][filter][block][reconfigure]") {
    Parser::Map sourceConfig;
    sourceConfig["signalDataType"] = std::string("CF32");
    sourceConfig["bufferSize"] = std::string("512");

    REQUIRE(flowgraph->blockCreate("src", "signal_generator", sourceConfig, {}) ==
            Result::SUCCESS);

    Parser::Map config;
    config["heads"] = std::string("4");
    config["center"] = std::string("[0, 100000, -100000, 200000]");

    TensorMap inputs;
    inputs["signal"].requested("src", "signal");

    REQUIRE(flowgraph->blockCreate("filter_shrink", "filter", config, inputs) ==
            Result::SUCCESS);
    REQUIRE(viewBlock("filter_shrink").state == Block::State::Created);

    Parser::Map update;
    update["sampleRate"] = std::string("2000000");
    update["bandwidth"] = std::string("1000000");
    update["taps"] = std::string("101");
    update["heads"] = std::string("1");
    update["center"] = std::string("[0, 100000, -100000, 200000]");

    REQUIRE(flowgraph->blockReconfigure("filter_shrink", update) == Result::SUCCESS);
    REQUIRE(viewBlock("filter_shrink").state == Block::State::Created);

    Parser::Map saved;
    REQUIRE(flowgraph->blockConfig("filter_shrink", saved) == Result::SUCCESS);
    REQUIRE(std::any_cast<U64>(saved.at("heads")) == 1);
    REQUIRE(std::any_cast<std::vector<F32>>(saved.at("center")).size() == 1);
}

TEST_CASE_METHOD(FlowgraphFixture,
                  "Filter block rejects invalid signal geometry before define",
                  "[modules][dsp][filter][block][validation]") {
    Blocks::OnesTensor source;
    source.shape = {1};
    source.dataType = "CF32";
    REQUIRE(flowgraph->blockCreate("geometry_src", source, {}) == Result::SUCCESS);

    Tensor sourceTensor = viewBlock("geometry_src").outputs.at("buffer").tensor;
    SECTION("rank zero") {
        REQUIRE(sourceTensor.squeezeDims(0) == Result::SUCCESS);
    }
    SECTION("zero last extent") {
        REQUIRE(sourceTensor.broadcastTo({0}) == Result::SUCCESS);
    }

    TensorMap inputs;
    inputs["signal"].requested("geometry_src", "buffer");
    REQUIRE(flowgraph->blockCreate("geometry_bad", Blocks::Filter{}, inputs) ==
            Result::SUCCESS);
    RequireRejectedBeforeDefine(viewBlock("geometry_bad"));
}

TEST_CASE_METHOD(FlowgraphFixture,
                  "Filter block rejects non-finite derived center bins before define",
                  "[modules][dsp][filter][block][validation]") {
    Blocks::OnesTensor source;
    source.shape = {512};
    source.dataType = "CF32";
    REQUIRE(flowgraph->blockCreate("center_src", source, {}) == Result::SUCCESS);

    Blocks::Filter config;
    SECTION("NaN bin") {
        config.center = {std::numeric_limits<F32>::quiet_NaN()};
    }
    SECTION("infinite bin") {
        config.center = {std::numeric_limits<F32>::infinity()};
    }

    TensorMap inputs;
    inputs["signal"].requested("center_src", "buffer");
    REQUIRE(flowgraph->blockCreate("center_bad", config, inputs) == Result::SUCCESS);
    RequireRejectedBeforeDefine(viewBlock("center_bad"));
}

TEST_CASE_METHOD(FlowgraphFixture,
                  "Filter block wraps negative centers during active resampling",
                  "[modules][dsp][filter][block][validation]") {
    Blocks::OnesTensor source;
    source.shape = {512};
    source.dataType = "CF32";
    REQUIRE(flowgraph->blockCreate("negative_src", source, {}) == Result::SUCCESS);

    Blocks::Filter config;
    config.center = {-100000.0f};

    TensorMap inputs;
    inputs["signal"].requested("negative_src", "buffer");
    REQUIRE(flowgraph->blockCreate("negative_filter", config, inputs) == Result::SUCCESS);
    const auto block = viewBlock("negative_filter");
    REQUIRE(block.state == Block::State::Created);
    REQUIRE(block.outputs.at("buffer").tensor.shape() == Shape{1, 256});
    REQUIRE(flowgraph->compute() == Result::SUCCESS);
}

TEST_CASE_METHOD(FlowgraphFixture,
                  "Filter block rejects combined convolution extent overflow before define",
                  "[modules][dsp][filter][block][validation]") {
    Blocks::OnesTensor source;
    source.shape = {512};
    source.dataType = "CF32";
    REQUIRE(flowgraph->blockCreate("overflow_src", source, {}) == Result::SUCCESS);

    Blocks::Filter config;
    config.taps = std::numeric_limits<U64>::max();

    TensorMap inputs;
    inputs["signal"].requested("overflow_src", "buffer");
    REQUIRE(flowgraph->blockCreate("overflow_filter", config, inputs) ==
            Result::SUCCESS);
    RequireRejectedBeforeDefine(viewBlock("overflow_filter"));
}

TEST_CASE_METHOD(FlowgraphFixture,
                  "Filter block safely bypasses an out-of-range resampler ratio",
                  "[modules][dsp][filter][block][validation]") {
    Blocks::OnesTensor source;
    source.shape = {512};
    source.dataType = "CF32";
    REQUIRE(flowgraph->blockCreate("ratio_src", source, {}) == Result::SUCCESS);

    Blocks::Filter config;
    config.sampleRate = std::numeric_limits<F32>::max();
    config.bandwidth = std::numeric_limits<F32>::min();

    TensorMap inputs;
    inputs["signal"].requested("ratio_src", "buffer");
    REQUIRE(flowgraph->blockCreate("ratio_filter", config, inputs) == Result::SUCCESS);
    const auto block = viewBlock("ratio_filter");
    REQUIRE(block.state == Block::State::Created);
    REQUIRE(block.outputs.at("buffer").tensor.shape() == Shape{1, 512});
    REQUIRE(flowgraph->compute() == Result::SUCCESS);
}

TEST_CASE_METHOD(FlowgraphFixture,
                  "Filter block safely delegates a huge finite center to Filter Taps",
                  "[modules][dsp][filter][block][validation]") {
    Blocks::OnesTensor source;
    source.shape = {512};
    source.dataType = "CF32";
    REQUIRE(flowgraph->blockCreate("huge_center_src", source, {}) == Result::SUCCESS);

    Blocks::Filter config;
    config.center = {std::numeric_limits<F32>::max()};

    TensorMap inputs;
    inputs["signal"].requested("huge_center_src", "buffer");
    REQUIRE(flowgraph->blockCreate("huge_center_filter", config, inputs) ==
            Result::SUCCESS);

    const auto block = viewBlock("huge_center_filter");
    REQUIRE(block.state == Block::State::Errored);
    REQUIRE_FALSE(block.interfaceInputs.empty());
    REQUIRE_FALSE(block.interfaceOutputs.empty());
    REQUIRE_FALSE(block.interfaceConfigs.empty());
    REQUIRE(block.outputs.empty());
}

TEST_CASE_METHOD(FlowgraphFixture,
                  "Filter block rolls back non-finite mapping before valid recreation",
                  "[modules][dsp][filter][block][reconfigure][validation]") {
    Blocks::OnesTensor source;
    source.shape = {512};
    source.dataType = "CF32";
    REQUIRE(flowgraph->blockCreate("rollback_src", source, {}) == Result::SUCCESS);

    Blocks::Filter config;
    TensorMap inputs;
    inputs["signal"].requested("rollback_src", "buffer");
    REQUIRE(flowgraph->blockCreate("rollback_filter", config, inputs) == Result::SUCCESS);
    REQUIRE(flowgraph->compute() == Result::SUCCESS);

    const Index initialOutputId =
        viewBlock("rollback_filter").outputs.at("buffer").tensor.id();

    Parser::Map invalidUpdate;
    invalidUpdate["center"] =
        std::vector<F32>{std::numeric_limits<F32>::quiet_NaN()};
    REQUIRE(flowgraph->blockReconfigure("rollback_filter", invalidUpdate) == Result::ERROR);

    auto block = viewBlock("rollback_filter");
    REQUIRE(block.state == Block::State::Created);
    REQUIRE(block.outputs.at("buffer").tensor.id() == initialOutputId);

    Parser::Map saved;
    REQUIRE(flowgraph->blockConfig("rollback_filter", saved) == Result::SUCCESS);
    REQUIRE(std::any_cast<std::vector<F32>>(saved.at("center")) == config.center);
    REQUIRE(flowgraph->compute() == Result::SUCCESS);

    Parser::Map validUpdate;
    validUpdate["center"] = std::vector<F32>{-100000.0f};
    REQUIRE(flowgraph->blockReconfigure("rollback_filter", validUpdate) ==
            Result::SUCCESS);

    block = viewBlock("rollback_filter");
    REQUIRE(block.state == Block::State::Created);
    REQUIRE(block.outputs.at("buffer").tensor.id() != initialOutputId);
    REQUIRE(flowgraph->compute() == Result::SUCCESS);

    saved.clear();
    REQUIRE(flowgraph->blockConfig("rollback_filter", saved) == Result::SUCCESS);
    REQUIRE(std::any_cast<std::vector<F32>>(saved.at("center")) ==
            std::vector<F32>{-100000.0f});
}
