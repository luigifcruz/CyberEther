#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <string>

#include "flowgraph_fixture.hh"
#include "jetstream/domains/core/ones_tensor/block.hh"
#include "jetstream/domains/dsp/decimator/block.hh"
#include "jetstream/domains/dsp/signal_generator/block.hh"

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

TEST_CASE("Decimator axis defaults to the last dimension",
          "[modules][dsp][decimator][config]") {
    REQUIRE(Blocks::Decimator{}.axis == -1);
}

TEST_CASE_METHOD(FlowgraphFixture,
                 "Decimator block reduces axis by ratio",
                 "[modules][dsp][decimator][block]") {
    Parser::Map sourceConfig;
    sourceConfig["signalDataType"] = std::string("CF32");
    sourceConfig["bufferSize"] = std::string("256");

    REQUIRE(flowgraph->blockCreate("src", "signal_generator", sourceConfig, {}) ==
            Result::SUCCESS);

    Parser::Map decimatorConfig;
    decimatorConfig["axis"] = std::string("0");
    decimatorConfig["ratio"] = std::string("4");

    TensorMap inputs;
    inputs["buffer"].requested("src", "signal");

    REQUIRE(flowgraph->blockCreate("decimator", "decimator", decimatorConfig, inputs) ==
            Result::SUCCESS);
    const auto block = viewBlock("decimator");
    REQUIRE(block.state == Block::State::Created);

    const auto axis = std::find_if(block.interfaceConfigs.begin(),
                                   block.interfaceConfigs.end(),
                                   [](const auto& entry) { return entry.name == "axis"; });
    REQUIRE(axis != block.interfaceConfigs.end());
    REQUIRE(axis->format == "int:");

    Parser::Map saved;
    REQUIRE(flowgraph->blockConfig("decimator", saved) == Result::SUCCESS);
    REQUIRE(saved.at("axis").type() == typeid(I64));
    REQUIRE(std::any_cast<I64>(saved.at("axis")) == 0);

    const Tensor out = block.outputs.at("buffer").tensor;
    REQUIRE(out.shape(0) == 64);
}

TEST_CASE_METHOD(FlowgraphFixture,
                 "Decimator block rejects invalid geometry before define",
                 "[modules][dsp][decimator][block][validation]") {
    Blocks::OnesTensor source;
    source.shape = {8, 6};
    source.dataType = "CF32";
    REQUIRE(flowgraph->blockCreate("geometry_src", source, {}) == Result::SUCCESS);

    TensorMap inputs;
    inputs["buffer"].requested("geometry_src", "buffer");

    Blocks::Decimator config;
    SECTION("zero ratio") {
        config.axis = 0;
        config.ratio = 0;
    }
    SECTION("positive out-of-range axis") {
        config.axis = 2;
        config.ratio = 2;
    }
    SECTION("negative out-of-range axis") {
        config.axis = -3;
        config.ratio = 2;
    }
    SECTION("indivisible axis extent") {
        config.axis = 1;
        config.ratio = 4;
    }

    REQUIRE(flowgraph->blockCreate("geometry_bad", config, inputs) == Result::SUCCESS);
    RequireRejectedBeforeDefine(viewBlock("geometry_bad"));
}

TEST_CASE_METHOD(FlowgraphFixture,
                 "Decimator block reduces a multidimensional negative axis",
                 "[modules][dsp][decimator][block][axis]") {
    Blocks::OnesTensor source;
    source.shape = {4, 3};
    source.dataType = "CF32";
    REQUIRE(flowgraph->blockCreate("src", source, {}) == Result::SUCCESS);

    Blocks::Decimator config;
    config.axis = -2;
    config.ratio = 2;

    TensorMap inputs;
    inputs["buffer"].requested("src", "buffer");
    REQUIRE(flowgraph->blockCreate("decimator", config, inputs) == Result::SUCCESS);
    REQUIRE(viewBlock("decimator").state == Block::State::Created);
    REQUIRE(flowgraph->compute() == Result::SUCCESS);

    const Tensor out = viewBlock("decimator").outputs.at("buffer").tensor;
    REQUIRE(out.shape() == Shape{2, 3});
    for (U64 index = 0; index < out.size(); ++index) {
        REQUIRE(out.data<CF32>()[index] == CF32(2.0f, 0.0f));
    }
}

TEST_CASE_METHOD(FlowgraphFixture,
                 "Decimator block remains incomplete without a resolved input",
                 "[modules][dsp][decimator][block][lifecycle]") {
    Blocks::Decimator config;
    config.axis = 8;
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

    Blocks::Decimator config;
    config.axis = 0;
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
        sourceConfig["frequency"] = F32{6000.0f};
        REQUIRE(flowgraph->blockCreate("metadata_src", "signal_generator", sourceConfig, {}) ==
                Result::SUCCESS);

        Blocks::Decimator config;
        config.axis = 0;
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
    }

    SECTION("absent sample rate remains absent while other attributes propagate") {
        Blocks::OnesTensor source;
        source.shape = {8};
        REQUIRE(flowgraph->blockCreate("no_rate_src", source, {}) == Result::SUCCESS);

        Tensor sourceTensor = viewBlock("no_rate_src").outputs.at("buffer").tensor;
        REQUIRE(sourceTensor.setAttribute("station", std::string("alpha")) == Result::SUCCESS);

        Blocks::Decimator config;
        config.axis = 0;
        config.ratio = 2;
        TensorMap inputs;
        inputs["buffer"].requested("no_rate_src", "buffer");
        REQUIRE(flowgraph->blockCreate("no_rate_decimator", config, inputs) ==
                Result::SUCCESS);

        const Tensor output = viewBlock("no_rate_decimator").outputs.at("buffer").tensor;
        REQUIRE_FALSE(output.hasAttribute("sampleRate"));
        REQUIRE(output.hasAttribute("station"));
        REQUIRE(std::any_cast<std::string>(output.attribute("station")) == "alpha");
    }

    SECTION("F64 sample rate is rejected before define") {
        Blocks::OnesTensor source;
        source.shape = {8};
        REQUIRE(flowgraph->blockCreate("f64_rate_src", source, {}) == Result::SUCCESS);

        Tensor sourceTensor = viewBlock("f64_rate_src").outputs.at("buffer").tensor;
        REQUIRE(sourceTensor.setAttribute("sampleRate", F64{48000.0}) == Result::SUCCESS);

        Blocks::Decimator config;
        config.axis = 0;
        config.ratio = 2;
        TensorMap inputs;
        inputs["buffer"].requested("f64_rate_src", "buffer");
        REQUIRE(flowgraph->blockCreate("f64_rate_decimator", config, inputs) ==
                Result::SUCCESS);
        RequireRejectedBeforeDefine(viewBlock("f64_rate_decimator"));
    }
}

TEST_CASE_METHOD(FlowgraphFixture,
                 "Decimator block recreates after a valid sparse axis update",
                 "[modules][dsp][decimator][block][reconfigure]") {
    Blocks::OnesTensor source;
    source.shape = {8, 6};
    source.dataType = "CF32";
    REQUIRE(flowgraph->blockCreate("recreate_src", source, {}) == Result::SUCCESS);

    Blocks::Decimator config;
    config.axis = 0;
    config.ratio = 2;
    TensorMap inputs;
    inputs["buffer"].requested("recreate_src", "buffer");
    REQUIRE(flowgraph->blockCreate("recreate_decimator", config, inputs) == Result::SUCCESS);
    REQUIRE(flowgraph->compute() == Result::SUCCESS);
    REQUIRE(viewBlock("recreate_decimator").outputs.at("buffer").tensor.shape() ==
            Shape{4, 6});

    Parser::Map update;
    update["axis"] = I64{1};
    REQUIRE(flowgraph->blockReconfigure("recreate_decimator", update) == Result::SUCCESS);

    const auto block = viewBlock("recreate_decimator");
    REQUIRE(block.state == Block::State::Created);
    REQUIRE(block.outputs.at("buffer").tensor.shape() == Shape{8, 3});

    Tensor output = block.outputs.at("buffer").tensor;
    std::fill(output.data<CF32>(), output.data<CF32>() + output.size(),
              CF32{-1.0f, -1.0f});
    REQUIRE(flowgraph->compute() == Result::SUCCESS);
    for (U64 index = 0; index < output.size(); ++index) {
        REQUIRE(output.data<CF32>()[index] == CF32(2.0f, 0.0f));
    }

    Parser::Map saved;
    REQUIRE(flowgraph->blockConfig("recreate_decimator", saved) == Result::SUCCESS);
    REQUIRE(std::any_cast<I64>(saved.at("axis")) == 1);
    REQUIRE(std::any_cast<U64>(saved.at("ratio")) == 2);
}

TEST_CASE_METHOD(FlowgraphFixture,
                 "Decimator block rolls back a rejected update before a later sparse update",
                 "[modules][dsp][decimator][block][reconfigure][validation]") {
    Blocks::SignalGenerator source;
    source.signalType = "dc";
    source.signalDataType = "CF32";
    source.bufferSize = 8;
    REQUIRE(flowgraph->blockCreate("rollback_src", source, {}) == Result::SUCCESS);

    Blocks::Decimator config;
    config.axis = 0;
    config.ratio = 2;
    TensorMap inputs;
    inputs["buffer"].requested("rollback_src", "signal");
    REQUIRE(flowgraph->blockCreate("rollback_decimator", config, inputs) == Result::SUCCESS);
    REQUIRE(flowgraph->compute() == Result::SUCCESS);
    const Index initialOutputId =
        viewBlock("rollback_decimator").outputs.at("buffer").tensor.id();

    Parser::Map invalidUpdate;
    invalidUpdate["ratio"] = U64{3};
    REQUIRE(flowgraph->blockReconfigure("rollback_decimator", invalidUpdate) == Result::ERROR);

    auto block = viewBlock("rollback_decimator");
    REQUIRE(block.state == Block::State::Created);
    REQUIRE(block.outputs.at("buffer").tensor.id() == initialOutputId);
    REQUIRE(block.outputs.at("buffer").tensor.shape() == Shape{4});

    Parser::Map saved;
    REQUIRE(flowgraph->blockConfig("rollback_decimator", saved) == Result::SUCCESS);
    REQUIRE(std::any_cast<I64>(saved.at("axis")) == 0);
    REQUIRE(std::any_cast<U64>(saved.at("ratio")) == 2);

    Tensor retainedOutput = block.outputs.at("buffer").tensor;
    std::fill(retainedOutput.data<CF32>(),
              retainedOutput.data<CF32>() + retainedOutput.size(),
              CF32{-1.0f, -1.0f});
    REQUIRE(flowgraph->compute() == Result::SUCCESS);
    for (U64 index = 0; index < retainedOutput.size(); ++index) {
        REQUIRE(retainedOutput.at<CF32>(index) == CF32(2.0f, 0.0f));
    }

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
    REQUIRE(std::any_cast<I64>(saved.at("axis")) == 0);
    REQUIRE(std::any_cast<U64>(saved.at("ratio")) == 4);
}
