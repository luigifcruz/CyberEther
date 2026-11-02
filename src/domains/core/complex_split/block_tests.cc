#include <catch2/catch_test_macros.hpp>

#include <string>
#include <vector>

#include "jetstream/domains/core/complex_split/block.hh"
#include "jetstream/domains/core/ones_tensor/block.hh"
#include "jetstream/domains/dsp/window/block.hh"
#include "flowgraph_fixture.hh"

using namespace Jetstream;

namespace {

std::vector<std::string> OutputLabels(const Flowgraph::View::BlockData& block) {
    std::vector<std::string> labels;
    for (const auto& entry : block.interfaceOutputs) {
        labels.push_back(entry.label);
    }
    return labels;
}

}  // namespace

TEST_CASE_METHOD(FlowgraphFixture, "Complex Split block creates and exposes outputs",
                 "[modules][complex_split][block]") {
    Blocks::Window source;
    source.size = 16;
    REQUIRE(flowgraph->blockCreate("split_src", source, {}) == Result::SUCCESS);

    TensorMap inputs;
    inputs["buffer"].requested("split_src", "window");

    Blocks::ComplexSplit config;
    REQUIRE(flowgraph->blockCreate("split_block", config, inputs) == Result::SUCCESS);

    const auto block = viewBlock("split_block");
    REQUIRE(block.state == Block::State::Created);
    REQUIRE(block.outputs.at("a").tensor.dtype() == DataType::F32);
    REQUIRE(block.outputs.at("b").tensor.dtype() == DataType::F32);
    REQUIRE(flowgraph->compute() == Result::SUCCESS);
}

TEST_CASE_METHOD(FlowgraphFixture, "Complex Split block switches mode",
                 "[modules][complex_split][block][reconfigure]") {
    Blocks::Window source;
    REQUIRE(flowgraph->blockCreate("split_mode_src", source, {}) == Result::SUCCESS);

    TensorMap inputs;
    inputs["buffer"].requested("split_mode_src", "window");

    Blocks::ComplexSplit config;
    REQUIRE(flowgraph->blockCreate("split_mode", config, inputs) == Result::SUCCESS);
    REQUIRE(OutputLabels(viewBlock("split_mode")) ==
            std::vector<std::string>{"Real", "Imag"});

    Parser::Map update;
    update["mode"] = std::string("polar");
    REQUIRE(flowgraph->blockReconfigure("split_mode", update) == Result::SUCCESS);
    REQUIRE(viewBlock("split_mode").state == Block::State::Created);
    REQUIRE(viewBlock("split_mode").outputs.count("a") == 1);
    REQUIRE(viewBlock("split_mode").outputs.count("b") == 1);
    REQUIRE(OutputLabels(viewBlock("split_mode")) ==
            std::vector<std::string>{"Magnitude", "Phase"});
    REQUIRE(flowgraph->compute() == Result::SUCCESS);

    Blocks::ComplexSplit polar;
    polar.mode = "polar";
    REQUIRE(flowgraph->blockCreate("split_polar", polar, inputs) == Result::SUCCESS);
    REQUIRE(OutputLabels(viewBlock("split_polar")) ==
            std::vector<std::string>{"Magnitude", "Phase"});
}

TEST_CASE_METHOD(FlowgraphFixture, "Complex Split block delegates real input to module validation",
                 "[modules][complex_split][block][validation]") {
    Blocks::OnesTensor source;
    source.shape = {8};
    REQUIRE(flowgraph->blockCreate("split_real_src", source, {}) == Result::SUCCESS);

    TensorMap inputs;
    inputs["buffer"].requested("split_real_src", "buffer");

    REQUIRE(flowgraph->blockCreate("split_real", "complex_split", {}, inputs) ==
            Result::SUCCESS);
    REQUIRE(viewBlock("split_real").state == Block::State::Errored);
    REQUIRE(viewBlock("split_real").outputs.empty());
}
