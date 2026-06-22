#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <any>
#include <filesystem>
#include <string>
#include <vector>

#include "flowgraph_fixture.hh"
#include "jetstream/domains/ml/onnx_inference/block.hh"

using namespace Jetstream;

namespace {

bool HasInterfaceKey(const std::vector<Flowgraph::View::InterfaceEntry>& entries,
                     const std::string& key) {
    return std::any_of(entries.begin(), entries.end(),
                       [&](const auto& entry) { return entry.name == key; });
}

}  // namespace

TEST_CASE_METHOD(FlowgraphFixture, "ONNX inference block exposes ports from tensor counts",
                 "[modules][onnx_inference][block]") {
    Blocks::OnnxInference config;
    config.modelInputCount = 2;
    config.modelOutputCount = 3;
    config.inputNames = {"input_a", "input_b"};
    config.outputNames = {"output_a", "output_b", "output_c"};

    REQUIRE(flowgraph->blockCreate("onnx_multi", config, {}) == Result::SUCCESS);
    const auto block = viewBlock("onnx_multi");
    REQUIRE(block.state == Block::State::Incomplete);

    REQUIRE(HasInterfaceKey(block.interfaceInputs, "input_0"));
    REQUIRE(HasInterfaceKey(block.interfaceInputs, "input_1"));
    REQUIRE_FALSE(HasInterfaceKey(block.interfaceInputs, "input"));

    REQUIRE(HasInterfaceKey(block.interfaceOutputs, "output_0"));
    REQUIRE(HasInterfaceKey(block.interfaceOutputs, "output_1"));
    REQUIRE(HasInterfaceKey(block.interfaceOutputs, "output_2"));
    REQUIRE_FALSE(HasInterfaceKey(block.interfaceOutputs, "output"));

    REQUIRE(HasInterfaceKey(block.interfaceConfigs, "modelInputCount"));
    REQUIRE(HasInterfaceKey(block.interfaceConfigs, "modelOutputCount"));
    REQUIRE(HasInterfaceKey(block.interfaceConfigs, "inputNames"));
    REQUIRE(HasInterfaceKey(block.interfaceConfigs, "outputNames"));
}

TEST_CASE_METHOD(FlowgraphFixture, "ONNX inference block uses counts over vector lengths",
                 "[modules][onnx_inference][block]") {
    Parser::Map config;
    config["inputNames"] = std::vector<std::string>{"input_a", "input_b"};
    config["outputNames"] = std::vector<std::string>{"output_a", "output_b"};

    REQUIRE(flowgraph->blockCreate("onnx_legacy", "onnx_inference", config, {}) == Result::SUCCESS);
    const auto block = viewBlock("onnx_legacy");
    REQUIRE(block.state == Block::State::Incomplete);

    REQUIRE(HasInterfaceKey(block.interfaceInputs, "input_0"));
    REQUIRE_FALSE(HasInterfaceKey(block.interfaceInputs, "input_1"));
    REQUIRE_FALSE(HasInterfaceKey(block.interfaceInputs, "input"));
    REQUIRE(HasInterfaceKey(block.interfaceOutputs, "output_0"));
    REQUIRE_FALSE(HasInterfaceKey(block.interfaceOutputs, "output_1"));
    REQUIRE_FALSE(HasInterfaceKey(block.interfaceOutputs, "output"));

    Parser::Map saved;
    REQUIRE(flowgraph->blockConfig("onnx_legacy", saved) == Result::SUCCESS);
    REQUIRE(std::any_cast<U64>(saved.at("modelInputCount")) == 1);
    REQUIRE(std::any_cast<U64>(saved.at("modelOutputCount")) == 1);
}

TEST_CASE_METHOD(FlowgraphFixture, "ONNX inference tensor counts can shrink vectors",
                 "[modules][onnx_inference][block][reconfigure]") {
    Blocks::OnnxInference config;
    config.modelInputCount = 2;
    config.modelOutputCount = 2;
    config.inputNames = {"input_a", "input_b"};
    config.outputNames = {"output_a", "output_b"};

    REQUIRE(flowgraph->blockCreate("onnx_shrink", config, {}) == Result::SUCCESS);
    REQUIRE(viewBlock("onnx_shrink").state == Block::State::Incomplete);

    Parser::Map update;
    update["modelInputCount"] = static_cast<U64>(1);
    update["modelOutputCount"] = static_cast<U64>(1);

    REQUIRE(flowgraph->blockReconfigure("onnx_shrink", update) == Result::SUCCESS);
    const auto block = viewBlock("onnx_shrink");
    REQUIRE(block.state == Block::State::Incomplete);

    REQUIRE(HasInterfaceKey(block.interfaceInputs, "input_0"));
    REQUIRE_FALSE(HasInterfaceKey(block.interfaceInputs, "input_1"));
    REQUIRE_FALSE(HasInterfaceKey(block.interfaceInputs, "input"));
    REQUIRE(HasInterfaceKey(block.interfaceOutputs, "output_0"));
    REQUIRE_FALSE(HasInterfaceKey(block.interfaceOutputs, "output_1"));
    REQUIRE_FALSE(HasInterfaceKey(block.interfaceOutputs, "output"));

    Parser::Map saved;
    REQUIRE(flowgraph->blockConfig("onnx_shrink", saved) == Result::SUCCESS);
    REQUIRE(std::any_cast<U64>(saved.at("modelInputCount")) == 1);
    REQUIRE(std::any_cast<U64>(saved.at("modelOutputCount")) == 1);
}

TEST_CASE_METHOD(FlowgraphFixture, "ONNX multi-IO example flowgraph imports",
                 "[modules][onnx_inference][block][flowgraph]") {
    const std::string path = "examples/flowgraphs/onnx-multi-io.yml";
    if (!std::filesystem::exists(path)) {
        SKIP("ONNX multi-IO example flowgraph is not available.");
    }

    REQUIRE(flowgraph->importFromFile(path) == Result::SUCCESS);

    const auto block = viewBlock("infer");
    REQUIRE(block.state == Block::State::Incomplete);
    REQUIRE(HasInterfaceKey(block.interfaceInputs, "input_0"));
    REQUIRE(HasInterfaceKey(block.interfaceInputs, "input_1"));
    REQUIRE(HasInterfaceKey(block.interfaceOutputs, "output_0"));
    REQUIRE(HasInterfaceKey(block.interfaceOutputs, "output_1"));
}

TEST_CASE_METHOD(FlowgraphFixture, "FRBNN ONNX example imports when local resources exist",
                 "[modules][onnx_inference][block][flowgraph][frbnn]") {
    if (!std::filesystem::exists("../stelline/resources/frbnn/model/frbnn_preprocessor.onnx") ||
        !std::filesystem::exists("../stelline/resources/frbnn/model/frbnn.onnx") ||
        !std::filesystem::exists("../cyberether-blocks/inference/frb_test_data.bin")) {
        SKIP("FRBNN sibling resources are not available.");
    }

    REQUIRE(flowgraph->importFromFile("examples/flowgraphs/frbnn-inference.yml") == Result::SUCCESS);

    const auto preprocessor = viewBlock("preprocessor");
    REQUIRE(preprocessor.state == Block::State::Created);
    REQUIRE(HasInterfaceKey(preprocessor.interfaceInputs, "input_0"));
    REQUIRE(HasInterfaceKey(preprocessor.interfaceOutputs, "output_0"));

    const auto frbnn = viewBlock("frbnn");
    REQUIRE(frbnn.state == Block::State::Created);
    REQUIRE(HasInterfaceKey(frbnn.interfaceInputs, "input_0"));
    REQUIRE(HasInterfaceKey(frbnn.interfaceOutputs, "output_0"));
}
