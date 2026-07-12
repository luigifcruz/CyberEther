#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <filesystem>
#include <string>
#include <vector>

#include "flowgraph_fixture.hh"
#include "jetstream/domains/ml/onnx_inference/block.hh"
#include "jetstream/platform.hh"
#include "jetstream/registry.hh"

using namespace Jetstream;

namespace {

bool HasInterfaceKey(const std::vector<Flowgraph::View::InterfaceEntry>& entries,
                     const std::string& key) {
    return std::any_of(entries.begin(), entries.end(),
                       [&](const auto& entry) { return entry.name == key; });
}

}  // namespace

TEST_CASE_METHOD(FlowgraphFixture, "ONNX inference block waits for a model before exposing tensor ports",
                 "[modules][onnx_inference][block]") {
    Blocks::OnnxInference config;

    REQUIRE(flowgraph->blockCreate("onnx_empty", config, {}) == Result::SUCCESS);
    const auto block = viewBlock("onnx_empty");
    REQUIRE(block.state == Block::State::Incomplete);

    REQUIRE(block.interfaceInputs.empty());
    REQUIRE(block.interfaceOutputs.empty());

    REQUIRE(HasInterfaceKey(block.interfaceConfigs, "modelPath"));
    REQUIRE(HasInterfaceKey(block.interfaceConfigs, "executionProvider"));

    REQUIRE_FALSE(HasInterfaceKey(block.interfaceConfigs, "modelInputCount"));
    REQUIRE_FALSE(HasInterfaceKey(block.interfaceConfigs, "modelOutputCount"));
    REQUIRE_FALSE(HasInterfaceKey(block.interfaceConfigs, "inputNames"));
    REQUIRE_FALSE(HasInterfaceKey(block.interfaceConfigs, "outputNames"));
}

TEST_CASE_METHOD(FlowgraphFixture, "ONNX inference block ignores legacy manual tensor fields",
                 "[modules][onnx_inference][block]") {
    Parser::Map config;
    config["modelInputCount"] = U64{2};
    config["modelOutputCount"] = U64{2};
    config["inputNames"] = std::vector<std::string>{"input_a", "input_b"};
    config["outputNames"] = std::vector<std::string>{"output_a", "output_b"};

    REQUIRE(flowgraph->blockCreate("onnx_legacy", "onnx_inference", config, {}) == Result::SUCCESS);
    const auto block = viewBlock("onnx_legacy");
    REQUIRE(block.state == Block::State::Incomplete);

    REQUIRE(block.interfaceInputs.empty());
    REQUIRE(block.interfaceOutputs.empty());
    REQUIRE_FALSE(HasInterfaceKey(block.interfaceInputs, "input_0"));
    REQUIRE_FALSE(HasInterfaceKey(block.interfaceOutputs, "output_0"));

    Parser::Map saved;
    REQUIRE(flowgraph->blockConfig("onnx_legacy", saved) == Result::SUCCESS);
    REQUIRE_FALSE(saved.contains("modelInputCount"));
    REQUIRE_FALSE(saved.contains("modelOutputCount"));
    REQUIRE_FALSE(saved.contains("inputNames"));
    REQUIRE_FALSE(saved.contains("outputNames"));
}

TEST_CASE_METHOD(FlowgraphFixture, "ONNX inference block keeps config UI when model metadata cannot be read",
                 "[modules][onnx_inference][block]") {
    const std::string missingModel = "missing-jetstream-onnx-model-for-test.onnx";
    REQUIRE_FALSE(std::filesystem::exists(Platform::PathFromUtf8(missingModel)));

    Parser::Map config;
    config["modelPath"] = missingModel;

    REQUIRE(flowgraph->blockCreate("onnx_missing", "onnx_inference", config, {}) == Result::SUCCESS);
    const auto block = viewBlock("onnx_missing");
    REQUIRE(block.state == Block::State::Incomplete);

    REQUIRE(block.interfaceInputs.empty());
    REQUIRE(block.interfaceOutputs.empty());

    REQUIRE(HasInterfaceKey(block.interfaceConfigs, "modelPath"));
    REQUIRE(HasInterfaceKey(block.interfaceConfigs, "executionProvider"));
}
