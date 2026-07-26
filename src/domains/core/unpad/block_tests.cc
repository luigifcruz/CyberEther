#include <catch2/catch_test_macros.hpp>

#include <algorithm>

#include "jetstream/domains/core/ones_tensor/block.hh"
#include "jetstream/domains/core/pad/block.hh"
#include "jetstream/domains/core/unpad/block.hh"
#include "jetstream/domains/dsp/signal_generator/block.hh"
#include "jetstream/domains/dsp/window/block.hh"
#include "flowgraph_fixture.hh"

using namespace Jetstream;

TEST_CASE_METHOD(FlowgraphFixture, "Unpad block creates and exposes both outputs",
                 "[modules][unpad][block]") {
    REQUIRE(Blocks::Unpad{}.axis == -1);

    Blocks::Window source;
    source.size = 8;
    REQUIRE(flowgraph->blockCreate("unpad_src", source, {}) == Result::SUCCESS);

    TensorMap padInputs;
    padInputs["unpadded"].requested("unpad_src", "window");
    Blocks::Pad padConfig;
    padConfig.size = 4;
    padConfig.axis = 0;
    REQUIRE(flowgraph->blockCreate("unpad_pad", padConfig, padInputs) == Result::SUCCESS);

    TensorMap inputs;
    inputs["padded"].requested("unpad_pad", "padded");
    Blocks::Unpad config;
    config.size = 4;
    config.axis = 0;
    REQUIRE(flowgraph->blockCreate("unpad_block", config, inputs) == Result::SUCCESS);
    const auto block = viewBlock("unpad_block");
    REQUIRE(block.state == Block::State::Created);
    REQUIRE(block.outputs.count("unpadded") == 1);
    REQUIRE(block.outputs.count("pad") == 1);

    const auto axis = std::find_if(block.interfaceConfigs.begin(),
                                   block.interfaceConfigs.end(),
                                   [](const auto& entry) { return entry.name == "axis"; });
    REQUIRE(axis != block.interfaceConfigs.end());
    REQUIRE(axis->format == "int:");

    Parser::Map saved;
    REQUIRE(flowgraph->blockConfig("unpad_block", saved) == Result::SUCCESS);
    REQUIRE(saved.at("axis").type() == typeid(I64));
    REQUIRE(std::any_cast<I64>(saved.at("axis")) == 0);
}

TEST_CASE_METHOD(FlowgraphFixture, "Unpad block rejects invalid axis",
                 "[modules][unpad][block][validation]") {
    Blocks::Window source;
    REQUIRE(flowgraph->blockCreate("unpad_bad_src", source, {}) == Result::SUCCESS);

    TensorMap inputs;
    inputs["padded"].requested("unpad_bad_src", "window");

    Blocks::Unpad config;
    config.size = 1;
    config.axis = 5;
    REQUIRE(flowgraph->blockCreate("unpad_bad", config, inputs) == Result::SUCCESS);
    REQUIRE(viewBlock("unpad_bad").state == Block::State::Errored);
    REQUIRE(viewBlock("unpad_bad").outputs.empty());
}

TEST_CASE_METHOD(FlowgraphFixture, "Unpad block delegates dtype validation to its module",
                 "[modules][unpad][block][validation]") {
    Blocks::OnesTensor source;
    source.shape = {4};
    source.dataType = "F64";
    REQUIRE(flowgraph->blockCreate("unpad_dtype_src", source, {}) == Result::SUCCESS);

    TensorMap inputs;
    inputs["padded"].requested("unpad_dtype_src", "buffer");

    Blocks::Unpad config;
    config.size = 1;
    REQUIRE(flowgraph->blockCreate("unpad_dtype", config, inputs) == Result::SUCCESS);
    REQUIRE(viewBlock("unpad_dtype").state == Block::State::Errored);
    REQUIRE(viewBlock("unpad_dtype").outputs.empty());
}

TEST_CASE_METHOD(FlowgraphFixture, "Unpad block preserves execution after rejected update",
                 "[modules][unpad][block][reconfigure][validation]") {
    Blocks::SignalGenerator source;
    source.signalType = "dc";
    source.bufferSize = 5;
    REQUIRE(flowgraph->blockCreate("unpad_update_src", source, {}) == Result::SUCCESS);

    TensorMap inputs;
    inputs["padded"].requested("unpad_update_src", "signal");

    Blocks::Unpad config;
    config.axis = 0;
    config.size = 2;
    REQUIRE(flowgraph->blockCreate("unpad_update", config, inputs) == Result::SUCCESS);
    REQUIRE(flowgraph->compute() == Result::SUCCESS);

    Parser::Map update;
    update["axis"] = I64{1};
    REQUIRE(flowgraph->blockReconfigure("unpad_update", update) == Result::ERROR);
    REQUIRE(viewBlock("unpad_update").state == Block::State::Created);

    Parser::Map saved;
    REQUIRE(flowgraph->blockConfig("unpad_update", saved) == Result::SUCCESS);
    REQUIRE(std::any_cast<I64>(saved.at("axis")) == config.axis);
    REQUIRE(std::any_cast<U64>(saved.at("size")) == config.size);

    Tensor unpadded = viewBlock("unpad_update").outputs.at("unpadded").tensor;
    Tensor pad = viewBlock("unpad_update").outputs.at("pad").tensor;
    std::fill(unpadded.data<F32>(), unpadded.data<F32>() + unpadded.size(), -1.0f);
    std::fill(pad.data<F32>(), pad.data<F32>() + pad.size(), -1.0f);

    REQUIRE(flowgraph->compute() == Result::SUCCESS);
    REQUIRE(unpadded.shape() == Shape{3});
    REQUIRE(pad.shape() == Shape{2});
    for (U64 index = 0; index < unpadded.size(); ++index) {
        REQUIRE(unpadded.data<F32>()[index] == 1.0f);
    }
    for (U64 index = 0; index < pad.size(); ++index) {
        REQUIRE(pad.data<F32>()[index] == 1.0f);
    }
}
