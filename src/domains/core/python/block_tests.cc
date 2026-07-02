#include <catch2/catch_test_macros.hpp>

#include <any>
#include <cmath>
#include <string>

#include "jetstream/block_interface.hh"
#include "jetstream/domains/core/python/block.hh"
#include "jetstream/logger.hh"
#include "jetstream/runtime_context.hh"
#include "flowgraph_fixture.hh"

using namespace Jetstream;

namespace {

bool OptionalPythonRuntimeUnavailableForBlock() {
    const auto& error = JST_LOG_LAST_ERROR();
    return error.find("Can't load Python library") != std::string::npos ||
           error.find("Can't initialize Python runtime helpers") != std::string::npos ||
           error.find("Can't load Python symbol") != std::string::npos ||
           error.find("Auto could not find a valid Python runtime") != std::string::npos ||
           error.find("No libpython was found") != std::string::npos ||
           error.find("No loadable libpython was found") != std::string::npos;
}

}  // namespace

TEST_CASE_METHOD(FlowgraphFixture,
                 "Python block exposes dynamic ports from counts",
                 "[modules][python][block]") {
    Blocks::Python config;
    config.inputCount = 2;
    config.outputCount = 3;

    REQUIRE(flowgraph->blockCreate("python_ports", config, {}, DeviceType::CPU, RuntimeType::PYTHON) ==
            Result::SUCCESS);

    const auto block = viewBlock("python_ports");
    REQUIRE(block.state == Block::State::Incomplete);
    REQUIRE(block.interfaceInputs.size() == 2);
    REQUIRE(block.interfaceOutputs.size() == 3);
    REQUIRE(block.interfaceConfigs.size() == 6);
    REQUIRE(block.interfaceInputs.at(0).name == "input0");
    REQUIRE(block.interfaceInputs.at(1).name == "input1");
    REQUIRE(block.interfaceOutputs.at(0).name == "output0");
    REQUIRE(block.interfaceOutputs.at(2).name == "output2");
}

TEST_CASE_METHOD(FlowgraphFixture,
                 "Python block can create and reconfigure a zero-port compute module",
                 "[modules][python][block]") {
    Blocks::Python config;
    config.code = "def compute(ctx):\n    pass\n";
    config.inputCount = 0;
    config.outputCount = 0;

    REQUIRE(flowgraph->blockCreate("python_zero", config, {}, DeviceType::CPU, RuntimeType::PYTHON) ==
            Result::SUCCESS);

    auto block = viewBlock("python_zero");
    if (block.state == Block::State::Errored && OptionalPythonRuntimeUnavailableForBlock()) {
        SUCCEED("Skipping Python block create test because the local Python runtime is unavailable.");
        return;
    }

    REQUIRE(block.state == Block::State::Created);
    REQUIRE(block.inputs.empty());
    REQUIRE(block.outputs.empty());

    Parser::Map nextConfig;
    nextConfig["code"] = std::string("def compute(ctx):\n    pass\n");
    nextConfig["inputCount"] = U64{0};
    nextConfig["outputCount"] = U64{0};

    REQUIRE(flowgraph->blockReconfigure("python_zero", nextConfig) == Result::SUCCESS);
    block = viewBlock("python_zero");
    REQUIRE(block.state == Block::State::Created);
}

TEST_CASE_METHOD(FlowgraphFixture,
                 "Python block creates a configured output without inputs",
                 "[modules][python][block]") {
    Blocks::Python config;
    config.code = "def compute(ctx):\n    ctx.outputs[0][...] = 3.0\n";
    config.inputCount = 0;
    config.outputCount = 1;
    config.outputTensorSpecs = {{.shape = "[4]"}};

    REQUIRE(flowgraph->blockCreate("python_source", config, {}, DeviceType::CPU, RuntimeType::PYTHON) ==
            Result::SUCCESS);

    auto block = viewBlock("python_source");
    if (block.state == Block::State::Errored && OptionalPythonRuntimeUnavailableForBlock()) {
        SUCCEED("Skipping Python block source test because the local Python runtime is unavailable.");
        return;
    }

    REQUIRE(block.state == Block::State::Created);
    REQUIRE(block.inputs.empty());
    REQUIRE(block.outputs.size() == 1);

    const Tensor& output = block.outputs.at("output0").tensor;
    REQUIRE(output.dtype() == DataType::F32);
    REQUIRE(output.shape() == Shape{4});

    REQUIRE(flowgraph->compute() == Result::SUCCESS);

    block = viewBlock("python_source");
    const Tensor& computed = block.outputs.at("output0").tensor;
    for (Index i = 0; i < computed.size(); ++i) {
        REQUIRE(std::abs(computed.at<F32>(i) - 3.0f) < 1e-5f);
    }
}

TEST_CASE_METHOD(FlowgraphFixture,
                 "Python block exposes console output through module context metric",
                 "[modules][python][block]") {
    Blocks::Python config;
    config.code = "def compute(ctx):\n    print(\"hello from metric\")\n";
    config.inputCount = 0;
    config.outputCount = 0;

    REQUIRE(flowgraph->blockCreate("python_console", config, {}, DeviceType::CPU, RuntimeType::PYTHON) ==
            Result::SUCCESS);

    auto block = viewBlock("python_console");
    if (block.state == Block::State::Errored && OptionalPythonRuntimeUnavailableForBlock()) {
        SUCCEED("Skipping Python block console test because the local Python runtime is unavailable.");
        return;
    }

    REQUIRE(block.state == Block::State::Created);
    REQUIRE(flowgraph->compute() == Result::SUCCESS);
    REQUIRE(flowgraph->view().metrics("python_console", block.metrics) == Result::SUCCESS);

    bool sawMetric = false;
    bool sawPrint = false;
    for (const auto& metric : block.metrics) {
        if (metric.format != "private-python-diagnostic" || !metric.value.has_value()) {
            continue;
        }

        sawMetric = true;
        const auto diagnostic = std::any_cast<Runtime::Context::Diagnostic>(metric.value);
        for (const auto& line : diagnostic.console) {
            sawPrint = sawPrint || line.find("hello from metric") != std::string::npos;
        }
    }

    REQUIRE(sawMetric);
    REQUIRE(sawPrint);
}

TEST_CASE_METHOD(FlowgraphFixture,
                 "Python block rejects malformed output tensor specs",
                 "[modules][python][block]") {
    Blocks::Python config;
    config.code = "def compute(ctx):\n    pass\n";
    config.inputCount = 0;
    config.outputCount = 1;
    config.outputTensorSpecs = {{.shape = "[1x]"}};

    REQUIRE(flowgraph->blockCreate("python_bad_tensor_spec", config, {}, DeviceType::CPU, RuntimeType::PYTHON) ==
            Result::SUCCESS);

    const auto block = viewBlock("python_bad_tensor_spec");
    REQUIRE(block.state == Block::State::Errored);
}
