#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <string>

#include "flowgraph_fixture.hh"
#include "jetstream/domains/core/ones_tensor/block.hh"
#include "jetstream/domains/dsp/decimator/block.hh"
#include "jetstream/domains/dsp/signal_generator/block.hh"

using namespace Jetstream;

namespace {

void RequireErroredWithInterface(const Flowgraph::View::BlockData& block) {
    REQUIRE(block.state == Block::State::Errored);
    REQUIRE_FALSE(block.interfaceInputs.empty());
    REQUIRE_FALSE(block.interfaceOutputs.empty());
    REQUIRE_FALSE(block.interfaceConfigs.empty());
    REQUIRE(block.outputs.empty());
}

}  // namespace

TEST_CASE_METHOD(FlowgraphFixture,
                  "Decimator block reduces samples by ratio",
                  "[modules][dsp][decimator][block]") {
    Parser::Map sourceConfig;
    sourceConfig["signalDataType"] = std::string("CF32");
    sourceConfig["bufferSize"] = std::string("256");

    REQUIRE(flowgraph->blockCreate("src", "signal_generator", sourceConfig, {}) ==
            Result::SUCCESS);
    Tensor sourceTensor = viewBlock("src").outputs.at("signal").tensor;
    REQUIRE(sourceTensor.setAttribute("sampleAxis", Index{0}) == Result::SUCCESS);

    Parser::Map decimatorConfig;
    decimatorConfig["ratio"] = std::string("4");

    TensorMap inputs;
    inputs["buffer"].requested("src", "signal");

    REQUIRE(flowgraph->blockCreate("decimator", "decimator", decimatorConfig, inputs) ==
            Result::SUCCESS);
    const auto block = viewBlock("decimator");
    REQUIRE(block.state == Block::State::Created);

    const auto ratio = std::find_if(block.interfaceConfigs.begin(),
                                    block.interfaceConfigs.end(),
                                    [](const auto& entry) { return entry.name == "ratio"; });
    REQUIRE(ratio != block.interfaceConfigs.end());
    REQUIRE(ratio->format == "uint:");

    Parser::Map saved;
    REQUIRE(flowgraph->blockConfig("decimator", saved) == Result::SUCCESS);
    REQUIRE(saved.at("ratio").type() == typeid(U64));
    REQUIRE(std::any_cast<U64>(saved.at("ratio")) == 4);

    const Tensor out = block.outputs.at("buffer").tensor;
    REQUIRE(out.shape(0) == 64);
    REQUIRE(std::any_cast<Index>(out.attribute("sampleAxis")) == 0);
    REQUIRE_FALSE(out.hasAttribute("batchAxis"));
    REQUIRE_FALSE(out.hasAttribute("channelAxis"));
}

TEST_CASE_METHOD(FlowgraphFixture,
                  "Decimator block rejects invalid ratio geometry before define",
                  "[modules][dsp][decimator][block][validation]") {
    Blocks::OnesTensor source;
    source.shape = {8, 6};
    source.dataType = "CF32";
    REQUIRE(flowgraph->blockCreate("geometry_src", source, {}) == Result::SUCCESS);
    Tensor sourceTensor = viewBlock("geometry_src").outputs.at("buffer").tensor;
    REQUIRE(sourceTensor.setAttribute("sampleAxis", Index{1}) == Result::SUCCESS);
    REQUIRE(sourceTensor.setAttribute("batchAxis", Index{0}) == Result::SUCCESS);

    TensorMap inputs;
    inputs["buffer"].requested("geometry_src", "buffer");

    Blocks::Decimator config;
    SECTION("zero ratio") {
        config.ratio = 0;
    }
    SECTION("indivisible sample extent") {
        config.ratio = 4;
    }

    REQUIRE(flowgraph->blockCreate("geometry_bad", config, inputs) == Result::SUCCESS);
    RequireErroredWithInterface(viewBlock("geometry_bad"));
}

TEST_CASE_METHOD(FlowgraphFixture,
                  "Decimator block rejects invalid signal metadata",
                  "[modules][dsp][decimator][block][validation]") {
    Blocks::OnesTensor source;
    source.shape = {8, 6};
    source.dataType = "CF32";

    SECTION("batch axis has the wrong type") {
        REQUIRE(flowgraph->blockCreate("metadata_bad_src", source, {}) ==
                Result::SUCCESS);
        Tensor tensor = viewBlock("metadata_bad_src").outputs.at("buffer").tensor;
        REQUIRE(tensor.setAttribute("sampleAxis", Index{1}) == Result::SUCCESS);
        REQUIRE(tensor.setAttribute("batchAxis", I64{0}) == Result::SUCCESS);
    }
    SECTION("batch axis is out of bounds") {
        REQUIRE(flowgraph->blockCreate("metadata_bad_src", source, {}) ==
                Result::SUCCESS);
        Tensor tensor = viewBlock("metadata_bad_src").outputs.at("buffer").tensor;
        REQUIRE(tensor.setAttribute("sampleAxis", Index{1}) == Result::SUCCESS);
        REQUIRE(tensor.setAttribute("batchAxis", Index{2}) == Result::SUCCESS);
    }
    SECTION("sample axis is missing") {
        REQUIRE(flowgraph->blockCreate("metadata_bad_src", source, {}) ==
                Result::SUCCESS);
    }
    SECTION("sample axis has the wrong type") {
        REQUIRE(flowgraph->blockCreate("metadata_bad_src", source, {}) ==
                Result::SUCCESS);
        Tensor tensor = viewBlock("metadata_bad_src").outputs.at("buffer").tensor;
        REQUIRE(tensor.setAttribute("sampleAxis", I64{1}) == Result::SUCCESS);
    }
    SECTION("sample and channel axes conflict") {
        REQUIRE(flowgraph->blockCreate("metadata_bad_src", source, {}) ==
                Result::SUCCESS);
        Tensor tensor = viewBlock("metadata_bad_src").outputs.at("buffer").tensor;
        REQUIRE(tensor.setAttribute("sampleAxis", Index{1}) == Result::SUCCESS);
        REQUIRE(tensor.setAttribute("channelAxis", Index{1}) == Result::SUCCESS);
    }

    Blocks::Decimator config;
    config.ratio = 2;

    TensorMap inputs;
    inputs["buffer"].requested("metadata_bad_src", "buffer");
    REQUIRE(flowgraph->blockCreate("metadata_bad", config, inputs) == Result::SUCCESS);
    RequireErroredWithInterface(viewBlock("metadata_bad"));
}

TEST_CASE_METHOD(FlowgraphFixture,
                  "Decimator preserves channel and batch roles",
                  "[modules][dsp][decimator][block][metadata]") {
    Blocks::OnesTensor source;
    source.dataType = "CF32";

    bool batched = false;
    Shape expectedShape;
    SECTION("channels and samples") {
        source.shape = {3, 8};
        expectedShape = {3, 4};
    }
    SECTION("opaque axis, channels, and samples") {
        source.shape = {2, 3, 8};
        expectedShape = {2, 3, 4};
    }
    SECTION("batch, heads, and samples") {
        source.shape = {2, 3, 8};
        expectedShape = {2, 3, 4};
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

    Blocks::Decimator config;
    config.ratio = 2;
    TensorMap inputs;
    inputs["buffer"].requested("heads_src", "buffer");
    REQUIRE(flowgraph->blockCreate("heads_decimator", config, inputs) ==
            Result::SUCCESS);
    REQUIRE(flowgraph->compute() == Result::SUCCESS);

    const Tensor output = viewBlock("heads_decimator").outputs.at("buffer").tensor;
    REQUIRE(output.shape() == expectedShape);
    REQUIRE(std::any_cast<Index>(output.attribute("sampleAxis")) == sampleAxis);
    REQUIRE(std::any_cast<Index>(output.attribute("channelAxis")) == channelAxis);
    if (batched) {
        REQUIRE(output.attribute("batchAxis").type() == typeid(Index));
        REQUIRE(std::any_cast<Index>(output.attribute("batchAxis")) == 0);
    } else {
        REQUIRE_FALSE(output.hasAttribute("batchAxis"));
    }
    for (U64 index = 0; index < output.size(); ++index) {
        REQUIRE(output.data<CF32>()[index] == CF32(2.0f, 0.0f));
    }
}

TEST_CASE_METHOD(FlowgraphFixture,
                  "Decimator uses explicit samples around leading and trailing batches",
                  "[modules][dsp][decimator][block][metadata]") {
    Blocks::OnesTensor source;
    source.dataType = "CF32";

    Index batchAxis = 0;
    Shape expectedShape;
    SECTION("leading batch axis") {
        source.shape = {3, 8};
        batchAxis = 0;
        expectedShape = {3, 4};
    }
    SECTION("trailing batch axis") {
        source.shape = {8, 3};
        batchAxis = 1;
        expectedShape = {4, 3};
    }

    REQUIRE(flowgraph->blockCreate("src", source, {}) == Result::SUCCESS);
    Tensor sourceTensor = viewBlock("src").outputs.at("buffer").tensor;
    const Index sampleAxis = batchAxis == 0 ? Index{1} : Index{0};
    REQUIRE(sourceTensor.setAttribute("sampleAxis", sampleAxis) == Result::SUCCESS);
    REQUIRE(sourceTensor.setAttribute("batchAxis", batchAxis) == Result::SUCCESS);

    Blocks::Decimator config;
    config.ratio = 2;

    TensorMap inputs;
    inputs["buffer"].requested("src", "buffer");
    REQUIRE(flowgraph->blockCreate("decimator", config, inputs) == Result::SUCCESS);
    REQUIRE(viewBlock("decimator").state == Block::State::Created);
    REQUIRE(flowgraph->compute() == Result::SUCCESS);

    const Tensor out = viewBlock("decimator").outputs.at("buffer").tensor;
    REQUIRE(out.shape() == expectedShape);
    REQUIRE(std::any_cast<Index>(out.attribute("sampleAxis")) == sampleAxis);
    REQUIRE(out.attribute("batchAxis").type() == typeid(Index));
    REQUIRE(std::any_cast<Index>(out.attribute("batchAxis")) == batchAxis);
    for (U64 index = 0; index < out.size(); ++index) {
        REQUIRE(out.data<CF32>()[index] == CF32(2.0f, 0.0f));
    }
}

TEST_CASE_METHOD(FlowgraphFixture,
                  "Decimator restores roles that follow the sample axis",
                  "[modules][dsp][decimator][block][metadata]") {
    Blocks::OnesTensor source;
    source.shape = {8, 3, 2};
    source.dataType = "CF32";
    REQUIRE(flowgraph->blockCreate("trailing_roles_src", source, {}) ==
            Result::SUCCESS);
    Tensor sourceTensor = viewBlock("trailing_roles_src").outputs.at("buffer").tensor;
    REQUIRE(sourceTensor.setAttribute("sampleAxis", Index{0}) == Result::SUCCESS);
    REQUIRE(sourceTensor.setAttribute("channelAxis", Index{1}) == Result::SUCCESS);
    REQUIRE(sourceTensor.setAttribute("batchAxis", Index{2}) == Result::SUCCESS);

    Blocks::Decimator config;
    config.ratio = 2;
    TensorMap inputs;
    inputs["buffer"].requested("trailing_roles_src", "buffer");
    REQUIRE(flowgraph->blockCreate("trailing_roles_decimator", config, inputs) ==
            Result::SUCCESS);
    REQUIRE(flowgraph->compute() == Result::SUCCESS);

    const Tensor output =
        viewBlock("trailing_roles_decimator").outputs.at("buffer").tensor;
    REQUIRE(output.shape() == Shape{4, 3, 2});
    REQUIRE(std::any_cast<Index>(output.attribute("sampleAxis")) == 0);
    REQUIRE(std::any_cast<Index>(output.attribute("channelAxis")) == 1);
    REQUIRE(std::any_cast<Index>(output.attribute("batchAxis")) == 2);
}

TEST_CASE_METHOD(FlowgraphFixture,
                  "Decimator block remains incomplete without a resolved input",
                  "[modules][dsp][decimator][block][lifecycle]") {
    Blocks::Decimator config;
    config.ratio = 2;

    SECTION("missing input") {
        REQUIRE(flowgraph->blockCreate("missing_input", config, {}) == Result::SUCCESS);
        const auto block = viewBlock("missing_input");
        REQUIRE(block.state == Block::State::Incomplete);
        REQUIRE_FALSE(block.interfaceInputs.empty());
        REQUIRE_FALSE(block.interfaceOutputs.empty());
    }

    SECTION("unresolved input") {
        REQUIRE(flowgraph->blockCreate("incomplete_source", Blocks::Decimator{}, {}) ==
                Result::SUCCESS);

        TensorMap inputs;
        inputs["buffer"].requested("incomplete_source", "buffer");
        REQUIRE(flowgraph->blockCreate("unresolved_input", config, inputs) ==
                Result::SUCCESS);
        const auto block = viewBlock("unresolved_input");
        REQUIRE(block.state == Block::State::Incomplete);
        REQUIRE_FALSE(block.interfaceInputs.empty());
        REQUIRE_FALSE(block.interfaceOutputs.empty());
    }
}

TEST_CASE_METHOD(FlowgraphFixture,
                 "Decimator block delegates dtype validation to Arithmetic",
                 "[modules][dsp][decimator][block][validation]") {
    Blocks::OnesTensor source;
    source.shape = {8};
    source.dataType = "F64";
    REQUIRE(flowgraph->blockCreate("dtype_src", source, {}) == Result::SUCCESS);
    Tensor sourceTensor = viewBlock("dtype_src").outputs.at("buffer").tensor;
    REQUIRE(sourceTensor.setAttribute("sampleAxis", Index{0}) == Result::SUCCESS);

    Blocks::Decimator config;
    config.ratio = 2;

    TensorMap inputs;
    inputs["buffer"].requested("dtype_src", "buffer");
    REQUIRE(flowgraph->blockCreate("dtype_bad", config, inputs) == Result::SUCCESS);

    const auto block = viewBlock("dtype_bad");
    REQUIRE(block.state == Block::State::Errored);
    REQUIRE_FALSE(block.interfaceInputs.empty());
    REQUIRE_FALSE(block.interfaceOutputs.empty());
    REQUIRE_FALSE(block.interfaceConfigs.empty());
    REQUIRE(block.outputs.empty());
}

TEST_CASE_METHOD(FlowgraphFixture,
                 "Decimator block preserves and adjusts metadata",
                 "[modules][dsp][decimator][block][metadata]") {
    SECTION("F32 sample rate is adjusted and unrelated attributes are preserved") {
        Parser::Map sourceConfig;
        sourceConfig["bufferSize"] = U64{8};
        sourceConfig["sampleRate"] = F32{48000.0f};
        REQUIRE(flowgraph->blockCreate("metadata_src", "signal_generator", sourceConfig, {}) ==
                Result::SUCCESS);
        Tensor sourceTensor = viewBlock("metadata_src").outputs.at("signal").tensor;
        REQUIRE(sourceTensor.setAttribute("sampleAxis", Index{0}) == Result::SUCCESS);
        REQUIRE(sourceTensor.setAttribute("frequency", F32{6000.0f}) ==
                Result::SUCCESS);

        Blocks::Decimator config;
        config.ratio = 4;
        TensorMap inputs;
        inputs["buffer"].requested("metadata_src", "signal");
        REQUIRE(flowgraph->blockCreate("metadata_decimator", config, inputs) ==
                Result::SUCCESS);

        const Tensor output = viewBlock("metadata_decimator").outputs.at("buffer").tensor;
        REQUIRE(output.hasAttribute("sampleRate"));
        REQUIRE(output.attribute("sampleRate").type() == typeid(F32));
        REQUIRE(std::any_cast<F32>(output.attribute("sampleRate")) == 12000.0f);
        REQUIRE(output.hasAttribute("frequency"));
        REQUIRE(std::any_cast<F32>(output.attribute("frequency")) == 6000.0f);
        REQUIRE(std::any_cast<Index>(output.attribute("sampleAxis")) == 0);
        REQUIRE_FALSE(output.hasAttribute("batchAxis"));
    }

    SECTION("absent sample rate remains absent while other attributes propagate") {
        Blocks::OnesTensor source;
        source.shape = {8};
        REQUIRE(flowgraph->blockCreate("no_rate_src", source, {}) == Result::SUCCESS);

        Tensor sourceTensor = viewBlock("no_rate_src").outputs.at("buffer").tensor;
        REQUIRE(sourceTensor.setAttribute("sampleAxis", Index{0}) == Result::SUCCESS);
        REQUIRE(sourceTensor.setAttribute("station", std::string("alpha")) == Result::SUCCESS);

        Blocks::Decimator config;
        config.ratio = 2;
        TensorMap inputs;
        inputs["buffer"].requested("no_rate_src", "buffer");
        REQUIRE(flowgraph->blockCreate("no_rate_decimator", config, inputs) ==
                Result::SUCCESS);

        const Tensor output = viewBlock("no_rate_decimator").outputs.at("buffer").tensor;
        REQUIRE_FALSE(output.hasAttribute("sampleRate"));
        REQUIRE(output.hasAttribute("station"));
        REQUIRE(std::any_cast<std::string>(output.attribute("station")) == "alpha");
        REQUIRE(std::any_cast<Index>(output.attribute("sampleAxis")) == 0);
    }

    SECTION("F64 sample rate is rejected before define") {
        Blocks::OnesTensor source;
        source.shape = {8};
        REQUIRE(flowgraph->blockCreate("f64_rate_src", source, {}) == Result::SUCCESS);

        Tensor sourceTensor = viewBlock("f64_rate_src").outputs.at("buffer").tensor;
        REQUIRE(sourceTensor.setAttribute("sampleAxis", Index{0}) == Result::SUCCESS);
        REQUIRE(sourceTensor.setAttribute("sampleRate", F64{48000.0}) == Result::SUCCESS);

        Blocks::Decimator config;
        config.ratio = 2;
        TensorMap inputs;
        inputs["buffer"].requested("f64_rate_src", "buffer");
        REQUIRE(flowgraph->blockCreate("f64_rate_decimator", config, inputs) ==
                Result::SUCCESS);
        RequireErroredWithInterface(viewBlock("f64_rate_decimator"));
    }
}

TEST_CASE_METHOD(FlowgraphFixture,
                  "Decimator block recreates after a valid ratio update",
                  "[modules][dsp][decimator][block][reconfigure]") {
    Blocks::OnesTensor source;
    source.shape = {3, 8};
    source.dataType = "CF32";
    REQUIRE(flowgraph->blockCreate("recreate_src", source, {}) == Result::SUCCESS);
    Tensor sourceTensor = viewBlock("recreate_src").outputs.at("buffer").tensor;
    REQUIRE(sourceTensor.setAttribute("sampleAxis", Index{1}) == Result::SUCCESS);
    REQUIRE(sourceTensor.setAttribute("batchAxis", Index{0}) == Result::SUCCESS);

    Blocks::Decimator config;
    config.ratio = 2;
    TensorMap inputs;
    inputs["buffer"].requested("recreate_src", "buffer");
    REQUIRE(flowgraph->blockCreate("recreate_decimator", config, inputs) == Result::SUCCESS);
    REQUIRE(flowgraph->compute() == Result::SUCCESS);
    REQUIRE(viewBlock("recreate_decimator").outputs.at("buffer").tensor.shape() ==
            Shape{3, 4});

    Parser::Map update;
    update["ratio"] = U64{4};
    REQUIRE(flowgraph->blockReconfigure("recreate_decimator", update) == Result::SUCCESS);

    const auto block = viewBlock("recreate_decimator");
    REQUIRE(block.state == Block::State::Created);
    REQUIRE(block.outputs.at("buffer").tensor.shape() == Shape{3, 2});
    REQUIRE(block.outputs.at("buffer").tensor.attribute("batchAxis").type() ==
            typeid(Index));
    REQUIRE(std::any_cast<Index>(
                block.outputs.at("buffer").tensor.attribute("batchAxis")) == 0);
    REQUIRE(std::any_cast<Index>(
                block.outputs.at("buffer").tensor.attribute("sampleAxis")) == 1);

    Tensor output = block.outputs.at("buffer").tensor;
    std::fill(output.data<CF32>(), output.data<CF32>() + output.size(),
              CF32{-1.0f, -1.0f});
    REQUIRE(flowgraph->compute() == Result::SUCCESS);
    for (U64 index = 0; index < output.size(); ++index) {
        REQUIRE(output.data<CF32>()[index] == CF32(4.0f, 0.0f));
    }

    Parser::Map saved;
    REQUIRE(flowgraph->blockConfig("recreate_decimator", saved) == Result::SUCCESS);
    REQUIRE(std::any_cast<U64>(saved.at("ratio")) == 4);
}

TEST_CASE_METHOD(FlowgraphFixture,
                  "Decimator block recovers from an invalid candidate with a sparse update",
                 "[modules][dsp][decimator][block][reconfigure][validation]") {
    Blocks::SignalGenerator source;
    source.signalType = "dc";
    source.signalDataType = "CF32";
    source.bufferSize = 8;
    REQUIRE(flowgraph->blockCreate("rollback_src", source, {}) == Result::SUCCESS);
    Tensor sourceTensor = viewBlock("rollback_src").outputs.at("signal").tensor;
    REQUIRE(sourceTensor.setAttribute("sampleAxis", Index{0}) == Result::SUCCESS);

    Blocks::Decimator config;
    config.ratio = 2;
    TensorMap inputs;
    inputs["buffer"].requested("rollback_src", "signal");
    REQUIRE(flowgraph->blockCreate("rollback_decimator", config, inputs) == Result::SUCCESS);
    REQUIRE(flowgraph->compute() == Result::SUCCESS);
    Parser::Map invalidUpdate;
    invalidUpdate["ratio"] = U64{3};
    REQUIRE(flowgraph->blockReconfigure("rollback_decimator", invalidUpdate) == Result::SUCCESS);

    auto block = viewBlock("rollback_decimator");
    REQUIRE(block.state == Block::State::Errored);
    REQUIRE(block.outputs.empty());
    REQUIRE_FALSE(block.interfaceOutputs.empty());

    Parser::Map saved;
    REQUIRE(flowgraph->blockConfig("rollback_decimator", saved) == Result::SUCCESS);
    REQUIRE(std::any_cast<U64>(saved.at("ratio")) == 3);

    Parser::Map validSparseUpdate;
    validSparseUpdate["ratio"] = U64{4};
    REQUIRE(flowgraph->blockReconfigure("rollback_decimator", validSparseUpdate) ==
            Result::SUCCESS);

    block = viewBlock("rollback_decimator");
    REQUIRE(block.state == Block::State::Created);
    REQUIRE(block.outputs.at("buffer").tensor.shape() == Shape{2});

    Tensor recreatedOutput = block.outputs.at("buffer").tensor;
    std::fill(recreatedOutput.data<CF32>(),
              recreatedOutput.data<CF32>() + recreatedOutput.size(),
              CF32{-1.0f, -1.0f});
    REQUIRE(flowgraph->compute() == Result::SUCCESS);
    for (U64 index = 0; index < recreatedOutput.size(); ++index) {
        REQUIRE(recreatedOutput.at<CF32>(index) == CF32(4.0f, 0.0f));
    }

    saved.clear();
    REQUIRE(flowgraph->blockConfig("rollback_decimator", saved) == Result::SUCCESS);
    REQUIRE(std::any_cast<U64>(saved.at("ratio")) == 4);
}
