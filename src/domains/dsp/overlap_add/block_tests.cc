#include <catch2/catch_test_macros.hpp>

#include <any>
#include <string>

#include "flowgraph_fixture.hh"
#include "jetstream/domains/core/ones_tensor/block.hh"
#include "jetstream/domains/dsp/overlap_add/block.hh"

using namespace Jetstream;

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
    Tensor buffer = viewBlock("buffer_src").outputs.at("signal").tensor;
    Tensor overlap = viewBlock("overlap_src").outputs.at("window").tensor;
    REQUIRE(buffer.setAttribute("sampleAxis", Index{0}) == Result::SUCCESS);
    REQUIRE(overlap.setAttribute("sampleAxis", Index{0}) == Result::SUCCESS);

    TensorMap inputs;
    inputs["buffer"].requested("buffer_src", "signal");
    inputs["overlap"].requested("overlap_src", "window");

    REQUIRE(flowgraph->blockCreate("overlap_add", "overlap_add", {}, inputs) ==
            Result::SUCCESS);
    const auto block = viewBlock("overlap_add");
    REQUIRE(block.state == Block::State::Created);
    REQUIRE(block.outputs.contains("buffer"));
    REQUIRE(std::any_cast<Index>(
                block.outputs.at("buffer").tensor.attribute("sampleAxis")) == 0);
}

TEST_CASE_METHOD(FlowgraphFixture,
                 "Overlap-add block delegates shape validation to its module",
                 "[modules][dsp][overlap_add][block][validation]") {
    Blocks::OnesTensor bufferSource;
    bufferSource.shape = {2};
    REQUIRE(flowgraph->blockCreate("overlap_bad_buffer", bufferSource, {}) ==
            Result::SUCCESS);
    Tensor buffer = viewBlock("overlap_bad_buffer").outputs.at("buffer").tensor;
    REQUIRE(buffer.setAttribute("sampleAxis", Index{0}) == Result::SUCCESS);

    Blocks::OnesTensor overlapSource;
    overlapSource.shape = {3};
    REQUIRE(flowgraph->blockCreate("overlap_bad_overlap", overlapSource, {}) ==
            Result::SUCCESS);
    Tensor overlap = viewBlock("overlap_bad_overlap").outputs.at("buffer").tensor;
    REQUIRE(overlap.setAttribute("sampleAxis", Index{0}) == Result::SUCCESS);

    TensorMap inputs;
    inputs["buffer"].requested("overlap_bad_buffer", "buffer");
    inputs["overlap"].requested("overlap_bad_overlap", "buffer");

    Blocks::OverlapAdd config;
    REQUIRE(flowgraph->blockCreate("overlap_bad", config, inputs) == Result::SUCCESS);
    REQUIRE(viewBlock("overlap_bad").state == Block::State::Errored);
    REQUIRE(viewBlock("overlap_bad").outputs.empty());
}
