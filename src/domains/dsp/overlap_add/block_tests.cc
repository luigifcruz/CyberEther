#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <string>

#include "flowgraph_fixture.hh"
#include "jetstream/domains/core/ones_tensor/block.hh"
#include "jetstream/domains/dsp/overlap_add/block.hh"
#include "jetstream/domains/dsp/overlap_add/module.hh"

using namespace Jetstream;

TEST_CASE("Overlap-add axis configs default to the last dimension",
          "[modules][dsp][overlap_add][config]") {
    REQUIRE(Blocks::OverlapAdd{}.axis == -1);
    REQUIRE(Modules::OverlapAdd{}.axis == -1);
}

TEST_CASE_METHOD(FlowgraphFixture,
                 "Overlap-add block creates with both required inputs",
                 "[modules][dsp][overlap_add][block]") {
    Parser::Map bufferConfig;
    bufferConfig["signalDataType"] = std::string("CF32");
    bufferConfig["bufferSize"] = std::string("128");

    Parser::Map overlapConfig;
    overlapConfig["size"] = std::string("16");

    REQUIRE(flowgraph->blockCreate("buffer_src", "signal_generator", bufferConfig, {}) ==
            Result::SUCCESS);
    REQUIRE(flowgraph->blockCreate("overlap_src", "window", overlapConfig, {}) ==
            Result::SUCCESS);

    TensorMap inputs;
    inputs["buffer"].requested("buffer_src", "signal");
    inputs["overlap"].requested("overlap_src", "window");

    REQUIRE(flowgraph->blockCreate("overlap_add", "overlap_add", {}, inputs) ==
            Result::SUCCESS);
    const auto block = viewBlock("overlap_add");
    REQUIRE(block.state == Block::State::Created);
    REQUIRE(block.outputs.contains("buffer"));

    const auto axis = std::find_if(block.interfaceConfigs.begin(),
                                   block.interfaceConfigs.end(),
                                   [](const auto& entry) { return entry.name == "axis"; });
    REQUIRE(axis != block.interfaceConfigs.end());
    REQUIRE(axis->format == "int:axis");

    Parser::Map saved;
    REQUIRE(flowgraph->blockConfig("overlap_add", saved) == Result::SUCCESS);
    REQUIRE(saved.at("axis").type() == typeid(I64));
    REQUIRE(std::any_cast<I64>(saved.at("axis")) == -1);
}

TEST_CASE_METHOD(FlowgraphFixture,
                 "Overlap-add block delegates shape validation to its module",
                 "[modules][dsp][overlap_add][block][validation]") {
    Blocks::OnesTensor bufferSource;
    bufferSource.shape = {2};
    REQUIRE(flowgraph->blockCreate("overlap_bad_buffer", bufferSource, {}) ==
            Result::SUCCESS);

    Blocks::OnesTensor overlapSource;
    overlapSource.shape = {3};
    REQUIRE(flowgraph->blockCreate("overlap_bad_overlap", overlapSource, {}) ==
            Result::SUCCESS);

    TensorMap inputs;
    inputs["buffer"].requested("overlap_bad_buffer", "buffer");
    inputs["overlap"].requested("overlap_bad_overlap", "buffer");

    Blocks::OverlapAdd config;
    config.axis = 0;
    REQUIRE(flowgraph->blockCreate("overlap_bad", config, inputs) == Result::SUCCESS);
    REQUIRE(viewBlock("overlap_bad").state == Block::State::Errored);
    REQUIRE(viewBlock("overlap_bad").outputs.empty());
}

TEST_CASE_METHOD(FlowgraphFixture,
                  "Overlap-add block preserves invalid axis for recovery",
                 "[modules][dsp][overlap_add][block][reconfigure][validation]") {
    Blocks::OnesTensor bufferSource;
    bufferSource.shape = {4};
    REQUIRE(flowgraph->blockCreate("overlap_update_buffer", bufferSource, {}) ==
            Result::SUCCESS);

    Blocks::OnesTensor overlapSource;
    overlapSource.shape = {2};
    REQUIRE(flowgraph->blockCreate("overlap_update_overlap", overlapSource, {}) ==
            Result::SUCCESS);

    TensorMap inputs;
    inputs["buffer"].requested("overlap_update_buffer", "buffer");
    inputs["overlap"].requested("overlap_update_overlap", "buffer");

    Blocks::OverlapAdd config;
    config.axis = 0;
    REQUIRE(flowgraph->blockCreate("overlap_update", config, inputs) ==
            Result::SUCCESS);
    REQUIRE(flowgraph->compute() == Result::SUCCESS);

    Parser::Map update;
    update["axis"] = I64{1};
    REQUIRE(flowgraph->blockReconfigure("overlap_update", update) == Result::SUCCESS);
    REQUIRE(viewBlock("overlap_update").state == Block::State::Errored);
    REQUIRE(viewBlock("overlap_update").outputs.empty());

    Parser::Map saved;
    REQUIRE(flowgraph->blockConfig("overlap_update", saved) == Result::SUCCESS);
    REQUIRE(std::any_cast<I64>(saved.at("axis")) == 1);

    Parser::Map recovery;
    recovery["axis"] = config.axis;
    REQUIRE(flowgraph->blockReconfigure("overlap_update", recovery) == Result::SUCCESS);
    REQUIRE(viewBlock("overlap_update").state == Block::State::Created);

    Tensor output = viewBlock("overlap_update").outputs.at("buffer").tensor;
    std::fill(output.data<F32>(), output.data<F32>() + output.size(), -1.0f);

    REQUIRE(flowgraph->compute() == Result::SUCCESS);
    REQUIRE(output.shape() == Shape{4});
    for (U64 index = 0; index < output.size(); ++index) {
        REQUIRE(output.at<F32>(index) >= 1.0f);
    }
}
