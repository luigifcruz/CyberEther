#include <catch2/catch_test_macros.hpp>

#include <string>
#include <vector>

#include "jetstream/domains/core/complex_merge/block.hh"
#include "jetstream/domains/core/ones_tensor/block.hh"
#include "flowgraph_fixture.hh"

using namespace Jetstream;

namespace {

std::vector<std::string> InputLabels(const Flowgraph::View::BlockData& block) {
    std::vector<std::string> labels;
    for (const auto& entry : block.interfaceInputs) {
        labels.push_back(entry.label);
    }
    return labels;
}

}  // namespace

TEST_CASE_METHOD(FlowgraphFixture, "Complex Merge block creates and exposes output",
                 "[modules][complex_merge][block]") {
    Blocks::OnesTensor source;
    source.shape = {16};
    REQUIRE(flowgraph->blockCreate("merge_src_a", source, {}) == Result::SUCCESS);
    REQUIRE(flowgraph->blockCreate("merge_src_b", source, {}) == Result::SUCCESS);

    TensorMap inputs;
    inputs["a"].requested("merge_src_a", "buffer");
    inputs["b"].requested("merge_src_b", "buffer");

    Blocks::ComplexMerge config;
    REQUIRE(flowgraph->blockCreate("merge_block", config, inputs) == Result::SUCCESS);

    const auto block = viewBlock("merge_block");
    REQUIRE(block.state == Block::State::Created);
    REQUIRE(block.outputs.at("buffer").tensor.dtype() == DataType::CF32);
    REQUIRE(block.outputs.at("buffer").tensor.shape() == Shape{16});
    REQUIRE(flowgraph->compute() == Result::SUCCESS);
}

TEST_CASE_METHOD(FlowgraphFixture, "Complex Merge block switches mode",
                 "[modules][complex_merge][block][reconfigure]") {
    Blocks::OnesTensor source;
    source.shape = {16};
    REQUIRE(flowgraph->blockCreate("merge_mode_a", source, {}) == Result::SUCCESS);
    REQUIRE(flowgraph->blockCreate("merge_mode_b", source, {}) == Result::SUCCESS);

    TensorMap inputs;
    inputs["a"].requested("merge_mode_a", "buffer");
    inputs["b"].requested("merge_mode_b", "buffer");

    Blocks::ComplexMerge config;
    REQUIRE(flowgraph->blockCreate("merge_mode", config, inputs) == Result::SUCCESS);
    REQUIRE(InputLabels(viewBlock("merge_mode")) ==
            std::vector<std::string>{"Real", "Imag"});

    Parser::Map update;
    update["mode"] = std::string("polar");
    REQUIRE(flowgraph->blockReconfigure("merge_mode", update) == Result::SUCCESS);
    REQUIRE(viewBlock("merge_mode").state == Block::State::Created);
    REQUIRE(InputLabels(viewBlock("merge_mode")) ==
            std::vector<std::string>{"Magnitude", "Phase"});
    REQUIRE(flowgraph->compute() == Result::SUCCESS);

    Blocks::ComplexMerge polar;
    polar.mode = "polar";
    REQUIRE(flowgraph->blockCreate("merge_polar", polar, inputs) == Result::SUCCESS);
    REQUIRE(InputLabels(viewBlock("merge_polar")) ==
            std::vector<std::string>{"Magnitude", "Phase"});
}

TEST_CASE_METHOD(FlowgraphFixture, "Complex Merge block handles disconnect and reconnect",
                 "[modules][complex_merge][block][lifecycle]") {
    Blocks::OnesTensor source;
    source.shape = {16};
    REQUIRE(flowgraph->blockCreate("merge_life_a", source, {}) == Result::SUCCESS);
    REQUIRE(flowgraph->blockCreate("merge_life_b", source, {}) == Result::SUCCESS);

    TensorMap inputs;
    inputs["a"].requested("merge_life_a", "buffer");
    inputs["b"].requested("merge_life_b", "buffer");
    REQUIRE(flowgraph->blockCreate("merge_life", "complex_merge", {}, inputs) ==
            Result::SUCCESS);

    REQUIRE(flowgraph->blockDisconnect("merge_life", "b") == Result::SUCCESS);
    REQUIRE(viewBlock("merge_life").state == Block::State::Incomplete);

    REQUIRE(flowgraph->blockConnect("merge_life", "b", "merge_life_b", "buffer") ==
            Result::SUCCESS);
    REQUIRE(viewBlock("merge_life").state == Block::State::Created);
}

TEST_CASE_METHOD(FlowgraphFixture, "Complex Merge block delegates mismatched shapes to module validation",
                 "[modules][complex_merge][block][validation]") {
    Blocks::OnesTensor sourceA;
    sourceA.shape = {16};
    REQUIRE(flowgraph->blockCreate("merge_bad_a", sourceA, {}) == Result::SUCCESS);

    Blocks::OnesTensor sourceB;
    sourceB.shape = {8};
    REQUIRE(flowgraph->blockCreate("merge_bad_b", sourceB, {}) == Result::SUCCESS);

    TensorMap inputs;
    inputs["a"].requested("merge_bad_a", "buffer");
    inputs["b"].requested("merge_bad_b", "buffer");

    REQUIRE(flowgraph->blockCreate("merge_bad", "complex_merge", {}, inputs) ==
            Result::SUCCESS);
    REQUIRE(viewBlock("merge_bad").state == Block::State::Errored);
    REQUIRE(viewBlock("merge_bad").outputs.empty());
}
