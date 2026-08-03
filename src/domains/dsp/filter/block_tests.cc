#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <any>
#include <cmath>
#include <limits>
#include <string>
#include <vector>

#include "flowgraph_fixture.hh"
#include "jetstream/domains/core/ones_tensor/block.hh"
#include "jetstream/domains/dsp/filter/block.hh"

using namespace Jetstream;

namespace {

void RequireErroredWithInterface(const Flowgraph::View::BlockData& block) {
    REQUIRE(block.state == Block::State::Errored);
    REQUIRE_FALSE(block.interfaceInputs.empty());
    REQUIRE_FALSE(block.interfaceOutputs.empty());
    REQUIRE_FALSE(block.interfaceConfigs.empty());
    REQUIRE(block.outputs.empty());
}

void TagSamples(Tensor tensor, const Index sampleAxis) {
    REQUIRE(tensor.setAttribute("sampleAxis", sampleAxis) == Result::SUCCESS);
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
    TagSamples(viewBlock("src").outputs.at("signal").tensor, 0);

    TensorMap inputs;
    inputs["signal"].requested("src", "signal");

    REQUIRE(flowgraph->blockCreate("filter", "filter", {}, inputs) == Result::SUCCESS);
    REQUIRE(viewBlock("filter").state == Block::State::Created);
    REQUIRE(viewBlock("filter").outputs.contains("buffer"));
}

TEST_CASE_METHOD(FlowgraphFixture,
                 "Filter block promotes F32 input to CF32",
                 "[modules][dsp][filter][block][F32]") {
    Blocks::OnesTensor source;
    source.shape = {512};
    source.dataType = "F32";
    REQUIRE(flowgraph->blockCreate("real_src", source, {}) == Result::SUCCESS);
    TagSamples(viewBlock("real_src").outputs.at("buffer").tensor, 0);

    TensorMap inputs;
    inputs["signal"].requested("real_src", "buffer");
    REQUIRE(flowgraph->blockCreate("real_filter", Blocks::Filter{}, inputs) ==
            Result::SUCCESS);

    const auto block = viewBlock("real_filter");
    REQUIRE(block.state == Block::State::Created);
    const Tensor output = block.outputs.at("buffer").tensor;
    REQUIRE(output.dtype() == DataType::CF32);
    REQUIRE(output.shape() == Shape{1, 256});
    REQUIRE(std::any_cast<Index>(output.attribute("channelAxis")) == 0);
    REQUIRE(std::any_cast<Index>(output.attribute("sampleAxis")) == 1);
    REQUIRE(flowgraph->compute() == Result::SUCCESS);
}

TEST_CASE_METHOD(FlowgraphFixture,
                  "Filter block rejects zero heads before define",
                  "[modules][dsp][filter][block][validation]") {
    Blocks::Filter config;
    config.heads = 0;

    REQUIRE(flowgraph->blockCreate("filter_zero_heads", config, {}) ==
            Result::SUCCESS);
    RequireErroredWithInterface(viewBlock("filter_zero_heads"));
}

TEST_CASE_METHOD(FlowgraphFixture,
                 "Filter block rejects unrepresentable heads without throwing",
                 "[modules][dsp][filter][block][validation]") {
    Blocks::Filter config;
    config.heads = std::numeric_limits<U64>::max();

    REQUIRE(flowgraph->blockCreate("filter_huge_heads", config, {}) ==
            Result::SUCCESS);
    RequireErroredWithInterface(viewBlock("filter_huge_heads"));
}

TEST_CASE_METHOD(FlowgraphFixture,
                 "Filter block uses heads over center length",
                 "[modules][dsp][filter][block]") {
    Parser::Map sourceConfig;
    sourceConfig["signalDataType"] = std::string("CF32");
    sourceConfig["bufferSize"] = std::string("512");

    REQUIRE(flowgraph->blockCreate("src", "signal_generator", sourceConfig, {}) ==
            Result::SUCCESS);
    TagSamples(viewBlock("src").outputs.at("signal").tensor, 0);

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
                  "Filter block creates multiple heads",
                  "[modules][dsp][filter][block]") {
    Blocks::OnesTensor source;
    source.shape = {512};
    source.dataType = "CF32";
    REQUIRE(flowgraph->blockCreate("multi_src", source, {}) == Result::SUCCESS);
    TagSamples(viewBlock("multi_src").outputs.at("buffer").tensor, 0);

    Blocks::Filter config;
    config.heads = 3;
    config.center = {0.0f, 400000.0f, -400000.0f};

    TensorMap inputs;
    inputs["signal"].requested("multi_src", "buffer");
    REQUIRE(flowgraph->blockCreate("multi_filter", config, inputs) == Result::SUCCESS);

    const auto block = viewBlock("multi_filter");
    REQUIRE(block.state == Block::State::Created);
    const Tensor output = block.outputs.at("buffer").tensor;
    REQUIRE(output.shape() == Shape{3, 256});
    REQUIRE(std::any_cast<Index>(output.attribute("channelAxis")) == 0);
    REQUIRE(std::any_cast<Index>(output.attribute("sampleAxis")) == 1);
    REQUIRE_FALSE(output.hasAttribute("batchAxis"));
    REQUIRE(flowgraph->compute() == Result::SUCCESS);
}

TEST_CASE_METHOD(FlowgraphFixture,
                  "Filter block heads can shrink stale center vector",
                  "[modules][dsp][filter][block][reconfigure]") {
    Parser::Map sourceConfig;
    sourceConfig["signalDataType"] = std::string("CF32");
    sourceConfig["bufferSize"] = std::string("512");

    REQUIRE(flowgraph->blockCreate("src", "signal_generator", sourceConfig, {}) ==
            Result::SUCCESS);
    TagSamples(viewBlock("src").outputs.at("signal").tensor, 0);

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
    TagSamples(sourceTensor, 0);
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
    RequireErroredWithInterface(viewBlock("geometry_bad"));
}

TEST_CASE_METHOD(FlowgraphFixture,
                  "Filter block rejects invalid signal metadata",
                  "[modules][dsp][filter][block][validation]") {
    Blocks::OnesTensor source;
    source.shape = {8, 2};
    source.dataType = "CF32";

    SECTION("batch axis has the wrong type") {
        REQUIRE(flowgraph->blockCreate("layout_src", source, {}) == Result::SUCCESS);
        Tensor tensor = viewBlock("layout_src").outputs.at("buffer").tensor;
        TagSamples(tensor, 0);
        REQUIRE(tensor.setAttribute("batchAxis", I64{1}) == Result::SUCCESS);
    }
    SECTION("batch axis is out of bounds") {
        REQUIRE(flowgraph->blockCreate("layout_src", source, {}) == Result::SUCCESS);
        Tensor tensor = viewBlock("layout_src").outputs.at("buffer").tensor;
        TagSamples(tensor, 0);
        REQUIRE(tensor.setAttribute("batchAxis", Index{2}) == Result::SUCCESS);
    }
    SECTION("sample axis is missing") {
        REQUIRE(flowgraph->blockCreate("layout_src", source, {}) == Result::SUCCESS);
    }
    SECTION("sample axis has the wrong type") {
        REQUIRE(flowgraph->blockCreate("layout_src", source, {}) == Result::SUCCESS);
        Tensor tensor = viewBlock("layout_src").outputs.at("buffer").tensor;
        REQUIRE(tensor.setAttribute("sampleAxis", I64{0}) == Result::SUCCESS);
    }
    SECTION("sample and batch roles conflict") {
        REQUIRE(flowgraph->blockCreate("layout_src", source, {}) == Result::SUCCESS);
        Tensor tensor = viewBlock("layout_src").outputs.at("buffer").tensor;
        TagSamples(tensor, 0);
        REQUIRE(tensor.setAttribute("batchAxis", Index{0}) == Result::SUCCESS);
    }
    TensorMap inputs;
    inputs["signal"].requested("layout_src", "buffer");
    REQUIRE(flowgraph->blockCreate("layout_bad", Blocks::Filter{}, inputs) ==
            Result::SUCCESS);
    RequireErroredWithInterface(viewBlock("layout_bad"));
}

TEST_CASE_METHOD(FlowgraphFixture,
                  "Filter block supports leading and trailing multi-head batches",
                  "[modules][dsp][filter][block][metadata]") {
    Blocks::OnesTensor source;
    source.dataType = "CF32";

    Index batchAxis = 0;
    Index expectedBatchAxis = 0;
    Shape expectedShape;
    SECTION("leading batch axis") {
        source.shape = {3, 4};
        batchAxis = 0;
        expectedBatchAxis = 0;
        expectedShape = {3, 2, 2};
    }
    SECTION("trailing batch axis") {
        source.shape = {4, 3};
        batchAxis = 1;
        expectedBatchAxis = 2;
        expectedShape = {2, 2, 3};
    }

    REQUIRE(flowgraph->blockCreate("batch_src", source, {}) == Result::SUCCESS);
    Tensor sourceTensor = viewBlock("batch_src").outputs.at("buffer").tensor;
    TagSamples(sourceTensor, batchAxis == 0 ? Index{1} : Index{0});
    REQUIRE(sourceTensor.setAttribute("batchAxis", batchAxis) == Result::SUCCESS);
    REQUIRE(sourceTensor.setAttribute("sampleRate", F32{8.0f}) == Result::SUCCESS);

    Blocks::Filter config;
    config.sampleRate = 8.0f;
    config.bandwidth = 4.0f;
    config.taps = 3;
    config.heads = 2;
    config.center = {0.0f, 1.0f};

    TensorMap inputs;
    inputs["signal"].requested("batch_src", "buffer");
    REQUIRE(flowgraph->blockCreate("batch_filter", config, inputs) == Result::SUCCESS);
    REQUIRE(flowgraph->compute() == Result::SUCCESS);

    const Tensor output = viewBlock("batch_filter").outputs.at("buffer").tensor;
    REQUIRE(output.shape() == expectedShape);
    REQUIRE(output.attribute("batchAxis").type() == typeid(Index));
    REQUIRE(std::any_cast<Index>(output.attribute("batchAxis")) == expectedBatchAxis);
    REQUIRE(std::any_cast<Index>(output.attribute("channelAxis")) ==
            (batchAxis == 0 ? Index{1} : Index{0}));
    REQUIRE(std::any_cast<Index>(output.attribute("sampleAxis")) ==
            (batchAxis == 0 ? Index{2} : Index{1}));
    REQUIRE(std::any_cast<F32>(output.attribute("sampleRate")) == 4.0f);
}

TEST_CASE_METHOD(FlowgraphFixture,
                  "Filter block rejects nested generated channels",
                  "[modules][dsp][filter][block][metadata][validation]") {
    Blocks::OnesTensor source;
    source.shape = {2, 8};
    source.dataType = "CF32";
    REQUIRE(flowgraph->blockCreate("channel_src", source, {}) == Result::SUCCESS);

    Tensor sourceTensor = viewBlock("channel_src").outputs.at("buffer").tensor;
    TagSamples(sourceTensor, 1);
    REQUIRE(sourceTensor.setAttribute("channelAxis", Index{0}) == Result::SUCCESS);

    TensorMap inputs;
    inputs["signal"].requested("channel_src", "buffer");
    REQUIRE(flowgraph->blockCreate("channel_filter", Blocks::Filter{}, inputs) ==
            Result::SUCCESS);
    RequireErroredWithInterface(viewBlock("channel_filter"));
}

TEST_CASE_METHOD(FlowgraphFixture,
                  "Filter block rejects non-finite derived center bins before define",
                  "[modules][dsp][filter][block][validation]") {
    Blocks::OnesTensor source;
    source.shape = {512};
    source.dataType = "CF32";
    REQUIRE(flowgraph->blockCreate("center_src", source, {}) == Result::SUCCESS);
    TagSamples(viewBlock("center_src").outputs.at("buffer").tensor, 0);

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
    RequireErroredWithInterface(viewBlock("center_bad"));
}

TEST_CASE_METHOD(FlowgraphFixture,
                  "Filter block wraps negative centers during active resampling",
                  "[modules][dsp][filter][block][validation]") {
    Blocks::OnesTensor source;
    source.shape = {512};
    source.dataType = "CF32";
    REQUIRE(flowgraph->blockCreate("negative_src", source, {}) == Result::SUCCESS);
    TagSamples(viewBlock("negative_src").outputs.at("buffer").tensor, 0);

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
    TagSamples(viewBlock("overflow_src").outputs.at("buffer").tensor, 0);

    Blocks::Filter config;
    config.taps = std::numeric_limits<U64>::max();

    TensorMap inputs;
    inputs["signal"].requested("overflow_src", "buffer");
    REQUIRE(flowgraph->blockCreate("overflow_filter", config, inputs) ==
            Result::SUCCESS);
    RequireErroredWithInterface(viewBlock("overflow_filter"));
}

TEST_CASE_METHOD(FlowgraphFixture,
                  "Filter block safely bypasses an out-of-range resampler ratio",
                  "[modules][dsp][filter][block][validation]") {
    Blocks::OnesTensor source;
    source.shape = {512};
    source.dataType = "CF32";
    REQUIRE(flowgraph->blockCreate("ratio_src", source, {}) == Result::SUCCESS);
    TagSamples(viewBlock("ratio_src").outputs.at("buffer").tensor, 0);

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
    TagSamples(viewBlock("huge_center_src").outputs.at("buffer").tensor, 0);

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
                   "Filter block preserves non-finite mapping before valid recovery",
                  "[modules][dsp][filter][block][reconfigure][validation]") {
    Blocks::OnesTensor source;
    source.shape = {512};
    source.dataType = "CF32";
    REQUIRE(flowgraph->blockCreate("rollback_src", source, {}) == Result::SUCCESS);
    TagSamples(viewBlock("rollback_src").outputs.at("buffer").tensor, 0);

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
    REQUIRE(flowgraph->blockReconfigure("rollback_filter", invalidUpdate) == Result::SUCCESS);

    auto block = viewBlock("rollback_filter");
    REQUIRE(block.state == Block::State::Errored);
    REQUIRE(block.outputs.empty());

    Parser::Map saved;
    REQUIRE(flowgraph->blockConfig("rollback_filter", saved) == Result::SUCCESS);
    const auto invalidCenter = std::any_cast<std::vector<F32>>(saved.at("center"));
    REQUIRE(invalidCenter.size() == 1);
    REQUIRE(std::isnan(invalidCenter.front()));

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

TEST_CASE_METHOD(FlowgraphFixture,
                 "Filter block normalizes folded inverse FFT output",
                 "[modules][dsp][filter][block][numeric]") {
    Blocks::OnesTensor source;
    source.shape = {4};
    source.dataType = "CF32";
    REQUIRE(flowgraph->blockCreate("normalize_src", source, {}) ==
            Result::SUCCESS);
    TagSamples(viewBlock("normalize_src").outputs.at("buffer").tensor, 0);

    Blocks::Filter config;
    config.sampleRate = 8.0f;
    config.bandwidth = 4.0f;
    config.taps = 3;

    TensorMap inputs;
    inputs["signal"].requested("normalize_src", "buffer");
    REQUIRE(flowgraph->blockCreate("normalize_filter", config, inputs) ==
            Result::SUCCESS);
    REQUIRE(flowgraph->compute() == Result::SUCCESS);

    const Tensor output =
        viewBlock("normalize_filter").outputs.at("buffer").tensor;
    REQUIRE(output.shape() == Shape{1, 2});
    REQUIRE_THAT(output.at<CF32>(0, 0).real(),
                 Catch::Matchers::WithinAbs(0.0f, 1e-5f));
    REQUIRE_THAT(output.at<CF32>(0, 1).real(),
                 Catch::Matchers::WithinAbs(0.5f, 1e-5f));
    REQUIRE_THAT(output.at<CF32>(0, 0).imag(),
                 Catch::Matchers::WithinAbs(0.0f, 1e-5f));
    REQUIRE_THAT(output.at<CF32>(0, 1).imag(),
                 Catch::Matchers::WithinAbs(0.0f, 1e-5f));
}
