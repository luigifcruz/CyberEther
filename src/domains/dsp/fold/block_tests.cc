#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <string>

#include "flowgraph_fixture.hh"
#include "jetstream/domains/core/ones_tensor/block.hh"
#include "jetstream/domains/dsp/fold/block.hh"
#include "jetstream/domains/dsp/fold/module.hh"
#include "jetstream/domains/dsp/signal_generator/block.hh"

using namespace Jetstream;

TEST_CASE("Fold axis configs default to the last dimension",
          "[modules][dsp][fold][config]") {
    REQUIRE(Blocks::Fold{}.axis == -1);
    REQUIRE(Modules::Fold{}.axis == -1);
}

TEST_CASE_METHOD(FlowgraphFixture,
                 "Fold block creates for valid axis and size",
                 "[modules][dsp][fold][block]") {
    Parser::Map sourceConfig;
    sourceConfig["signalDataType"] = std::string("F32");
    sourceConfig["bufferSize"] = std::string("64");

    REQUIRE(flowgraph->blockCreate("src", "signal_generator", sourceConfig, {}) ==
            Result::SUCCESS);

    Parser::Map foldConfig;
    foldConfig["axis"] = std::string("0");
    foldConfig["offset"] = std::string("0");
    foldConfig["size"] = std::string("16");

    TensorMap inputs;
    inputs["buffer"].requested("src", "signal");

    REQUIRE(flowgraph->blockCreate("fold", "fold", foldConfig, inputs) == Result::SUCCESS);
    const auto block = viewBlock("fold");
    REQUIRE(block.state == Block::State::Created);

    const auto axis = std::find_if(block.interfaceConfigs.begin(),
                                   block.interfaceConfigs.end(),
                                   [](const auto& entry) { return entry.name == "axis"; });
    REQUIRE(axis != block.interfaceConfigs.end());
    REQUIRE(axis->format == "int:axis");

    Parser::Map saved;
    REQUIRE(flowgraph->blockConfig("fold", saved) == Result::SUCCESS);
    REQUIRE(saved.at("axis").type() == typeid(I64));
    REQUIRE(std::any_cast<I64>(saved.at("axis")) == 0);

    const Tensor out = block.outputs.at("buffer").tensor;
    REQUIRE(out.shape(0) == 16);
}

TEST_CASE_METHOD(FlowgraphFixture,
                 "Fold block delegates config validation to its module",
                 "[modules][dsp][fold][block][validation]") {
    Blocks::SignalGenerator source;
    source.bufferSize = 8;
    REQUIRE(flowgraph->blockCreate("fold_bad_src", source, {}) == Result::SUCCESS);

    TensorMap inputs;
    inputs["buffer"].requested("fold_bad_src", "signal");

    Blocks::Fold config;
    config.axis = 0;
    config.size = 3;
    REQUIRE(flowgraph->blockCreate("fold_bad", config, inputs) == Result::SUCCESS);
    REQUIRE(viewBlock("fold_bad").state == Block::State::Errored);
    REQUIRE(viewBlock("fold_bad").outputs.empty());
}

TEST_CASE_METHOD(FlowgraphFixture,
                 "Fold block delegates dtype validation to its module",
                 "[modules][dsp][fold][block][validation]") {
    Blocks::OnesTensor source;
    source.shape = {8};
    source.dataType = "F64";
    REQUIRE(flowgraph->blockCreate("fold_dtype_src", source, {}) == Result::SUCCESS);

    TensorMap inputs;
    inputs["buffer"].requested("fold_dtype_src", "buffer");

    Blocks::Fold config;
    config.axis = 0;
    config.size = 4;
    REQUIRE(flowgraph->blockCreate("fold_dtype", config, inputs) == Result::SUCCESS);
    REQUIRE(viewBlock("fold_dtype").state == Block::State::Errored);
    REQUIRE(viewBlock("fold_dtype").outputs.empty());
}

TEST_CASE_METHOD(FlowgraphFixture,
                  "Fold block preserves invalid size for recovery",
                 "[modules][dsp][fold][block][reconfigure][validation]") {
    Blocks::SignalGenerator source;
    source.signalType = "dc";
    source.bufferSize = 8;
    REQUIRE(flowgraph->blockCreate("fold_update_src", source, {}) == Result::SUCCESS);

    TensorMap inputs;
    inputs["buffer"].requested("fold_update_src", "signal");

    Blocks::Fold config;
    config.axis = 0;
    config.size = 4;
    REQUIRE(flowgraph->blockCreate("fold_update", config, inputs) == Result::SUCCESS);
    REQUIRE(flowgraph->compute() == Result::SUCCESS);

    Parser::Map update;
    update["size"] = U64{3};
    REQUIRE(flowgraph->blockReconfigure("fold_update", update) == Result::SUCCESS);
    REQUIRE(viewBlock("fold_update").state == Block::State::Errored);
    REQUIRE(viewBlock("fold_update").outputs.empty());

    Parser::Map saved;
    REQUIRE(flowgraph->blockConfig("fold_update", saved) == Result::SUCCESS);
    REQUIRE(std::any_cast<U64>(saved.at("size")) == 3);

    Parser::Map recovery;
    recovery["size"] = config.size;
    REQUIRE(flowgraph->blockReconfigure("fold_update", recovery) == Result::SUCCESS);
    REQUIRE(viewBlock("fold_update").state == Block::State::Created);

    Tensor output = viewBlock("fold_update").outputs.at("buffer").tensor;
    std::fill(output.data<F32>(), output.data<F32>() + output.size(), -1.0f);

    REQUIRE(flowgraph->compute() == Result::SUCCESS);
    REQUIRE(output.shape() == Shape{4});
    for (U64 index = 0; index < output.size(); ++index) {
        REQUIRE(output.at<F32>(index) == 1.0f);
    }
}
