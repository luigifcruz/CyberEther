#include <catch2/catch_test_macros.hpp>

#include <any>
#include <cmath>
#include <memory>
#include <string>
#include <unordered_set>
#include <vector>

#include "jetstream/domains/core/python/module.hh"
#include "jetstream/logger.hh"
#include "jetstream/module_context.hh"
#include "jetstream/registry.hh"
#include "jetstream/runtime.hh"
#include "jetstream/runtime_context.hh"

using namespace Jetstream;

namespace {

bool OptionalPythonRuntimeUnavailable() {
    const auto& error = JST_LOG_LAST_ERROR();
    return error.find("Can't load Python library") != std::string::npos ||
           error.find("Can't initialize Python runtime helpers") != std::string::npos ||
           error.find("Can't load Python symbol") != std::string::npos ||
           error.find("Auto could not find a valid Python runtime") != std::string::npos ||
           error.find("No libpython was found") != std::string::npos ||
           error.find("No loadable libpython was found") != std::string::npos;
}

void ConfigureF32Output(Modules::Python& config, const std::string& shape) {
    config.outputTensorSpecs = {{.shape = shape}};
}

}  // namespace

TEST_CASE("Python module runs configured compute code", "[modules][python][module]") {
    std::shared_ptr<Module> module;
    REQUIRE(Registry::BuildModule("python", DeviceType::CPU, RuntimeType::PYTHON, "generic", module) ==
            Result::SUCCESS);

    Tensor input;
    REQUIRE(input.create(DeviceType::CPU, DataType::F32, {4}) == Result::SUCCESS);
    for (Index i = 0; i < input.size(); ++i) {
        input.at<F32>(i) = static_cast<F32>(i + 1);
    }

    TensorMap inputs;
    inputs["input0"].produced("source", "output", input);

    Modules::Python config;
    config.code = R"PY(def compute(ctx):
    print("hello from python")
    ctx.outputs[0][...] = ctx.inputs[0] * 2.0
)PY";
    config.inputCount = 1;
    config.outputCount = 1;
    ConfigureF32Output(config, "[4]");

    const auto createResult = module->create("python_module", config, inputs);
    if (createResult != Result::SUCCESS && OptionalPythonRuntimeUnavailable()) {
        SUCCEED("Skipping Python module test because the local Python runtime is unavailable.");
        return;
    }

    REQUIRE(createResult == Result::SUCCESS);

    Runtime runtime("python", DeviceType::CPU, RuntimeType::PYTHON);
    REQUIRE(runtime.create({{"python_module", module}}) == Result::SUCCESS);

    std::unordered_set<std::string> skippedModules;
    std::unordered_set<std::string> failedModules;
    REQUIRE(runtime.compute({"python_module"}, skippedModules, failedModules) == Result::SUCCESS);

    const Tensor& output = module->outputs().at("output0").tensor;
    const auto* data = output.data<F32>();
    REQUIRE(data != nullptr);

    for (Index i = 0; i < output.size(); ++i) {
        REQUIRE(std::abs(data[i] - static_cast<F32>((i + 1) * 2)) < 1e-5f);
    }

    const auto diagnostic = module->context()->runtime()->diagnostic();
    bool sawPrint = false;
    for (const auto& line : diagnostic.console) {
        sawPrint = sawPrint || line.find("hello from python") != std::string::npos;
    }
    REQUIRE(sawPrint);

    REQUIRE(runtime.destroy() == Result::SUCCESS);
    REQUIRE(module->destroy() == Result::SUCCESS);
}

TEST_CASE("Python module exposes non-contiguous input tensors", "[modules][python][module]") {
    std::shared_ptr<Module> module;
    REQUIRE(Registry::BuildModule("python", DeviceType::CPU, RuntimeType::PYTHON, "generic", module) ==
            Result::SUCCESS);

    Tensor input;
    REQUIRE(input.create(DeviceType::CPU, DataType::F32, {2, 4}) == Result::SUCCESS);
    auto* inputData = input.data<F32>();
    REQUIRE(inputData != nullptr);
    for (Index i = 0; i < input.size(); ++i) {
        inputData[i] = static_cast<F32>(i + 1);
    }
    REQUIRE(input.permute({1, 0}) == Result::SUCCESS);
    REQUIRE_FALSE(input.contiguous());

    TensorMap inputs;
    inputs["input0"].produced("source", "output", input);

    Modules::Python config;
    config.code = R"PY(def compute(ctx):
    ctx.outputs[0][...] = ctx.inputs[0] * 2.0
)PY";
    config.inputCount = 1;
    config.outputCount = 1;
    ConfigureF32Output(config, "[4, 2]");

    const auto createResult = module->create("python_strided", config, inputs);
    if (createResult != Result::SUCCESS && OptionalPythonRuntimeUnavailable()) {
        SUCCEED("Skipping Python strided-input test because the local Python runtime is unavailable.");
        return;
    }

    REQUIRE(createResult == Result::SUCCESS);

    Runtime runtime("python", DeviceType::CPU, RuntimeType::PYTHON);
    REQUIRE(runtime.create({{"python_strided", module}}) == Result::SUCCESS);

    std::unordered_set<std::string> skippedModules;
    std::unordered_set<std::string> failedModules;
    REQUIRE(runtime.compute({"python_strided"}, skippedModules, failedModules) == Result::SUCCESS);
    REQUIRE(skippedModules.empty());

    const Tensor& output = module->outputs().at("output0").tensor;
    const auto* data = output.data<F32>();
    REQUIRE(data != nullptr);

    for (Index i = 0; i < 4; ++i) {
        for (Index j = 0; j < 2; ++j) {
            const F32 expected = static_cast<F32>(j * 4 + i + 1) * 2.0f;
            REQUIRE(std::abs(data[i * 2 + j] - expected) < 1e-5f);
        }
    }

    REQUIRE(runtime.destroy() == Result::SUCCESS);
    REQUIRE(module->destroy() == Result::SUCCESS);
}

TEST_CASE("Python module exposes tensor attributes", "[modules][python][module]") {
    std::shared_ptr<Module> module;
    REQUIRE(Registry::BuildModule("python", DeviceType::CPU, RuntimeType::PYTHON, "generic", module) ==
            Result::SUCCESS);

    Tensor input;
    REQUIRE(input.create(DeviceType::CPU, DataType::F32, {4}) == Result::SUCCESS);
    REQUIRE(input.setAttribute("sampleRate", static_cast<F32>(2000000.0f)) == Result::SUCCESS);
    REQUIRE(input.setAttribute("station", std::string("alpha")) == Result::SUCCESS);

    TensorMap inputs;
    inputs["input0"].produced("source", "output", input);

    Modules::Python config;
    config.code = R"PY(def compute(ctx):
    ctx.outputs[0][...] = ctx.inputs[0]
    ctx.output_attrs[0]["sampleRate"] = ctx.input_attrs[0]["sampleRate"] * 0.5
    ctx.output_attrs[0]["station"] = ctx.input_attrs[0]["station"] + "-out"
    ctx.output_attrs[0]["decimated"] = True
)PY";
    config.inputCount = 1;
    config.outputCount = 1;
    ConfigureF32Output(config, "[4]");

    const auto createResult = module->create("python_attrs", config, inputs);
    if (createResult != Result::SUCCESS && OptionalPythonRuntimeUnavailable()) {
        SUCCEED("Skipping Python attribute test because the local Python runtime is unavailable.");
        return;
    }

    REQUIRE(createResult == Result::SUCCESS);

    Runtime runtime("python", DeviceType::CPU, RuntimeType::PYTHON);
    REQUIRE(runtime.create({{"python_attrs", module}}) == Result::SUCCESS);

    std::unordered_set<std::string> skippedModules;
    std::unordered_set<std::string> failedModules;
    REQUIRE(runtime.compute({"python_attrs"}, skippedModules, failedModules) == Result::SUCCESS);
    REQUIRE(skippedModules.empty());

    const auto requireHalvedRate = [](const std::any& value) {
        if (value.type() == typeid(F32)) {
            REQUIRE(std::abs(std::any_cast<F32>(value) - 1000000.0f) < 1e-3f);
            return;
        }
        REQUIRE(std::abs(std::any_cast<F64>(value) - 1000000.0) < 1e-3);
    };

    const Tensor& output = module->outputs().at("output0").tensor;
    REQUIRE(output.hasAttribute("sampleRate"));
    requireHalvedRate(output.attribute("sampleRate"));
    REQUIRE(output.hasAttribute("station"));
    REQUIRE(std::any_cast<std::string>(output.attribute("station")) == "alpha-out");
    REQUIRE(output.hasAttribute("decimated"));
    REQUIRE(std::any_cast<bool>(output.attribute("decimated")) == true);

    REQUIRE(runtime.compute({"python_attrs"}, skippedModules, failedModules) == Result::SUCCESS);
    requireHalvedRate(output.attribute("sampleRate"));

    REQUIRE(runtime.destroy() == Result::SUCCESS);
    REQUIRE(module->destroy() == Result::SUCCESS);
}

TEST_CASE("Python module publishes nested attribute mutations", "[modules][python][module]") {
    std::shared_ptr<Module> module;
    REQUIRE(Registry::BuildModule("python", DeviceType::CPU, RuntimeType::PYTHON, "generic", module) ==
            Result::SUCCESS);

    Tensor input;
    REQUIRE(input.create(DeviceType::CPU, DataType::F32, {1}) == Result::SUCCESS);

    TensorMap inputs;
    inputs["input0"].produced("source", "output", input);

    Modules::Python config;
    config.code = R"PY(_n = 0

def compute(ctx):
    global _n
    _n += 1
    ctx.outputs[0][...] = ctx.inputs[0]
    if _n == 1:
        ctx.output_attrs[0]["meta"] = {"stage": 1}
    else:
        ctx.output_attrs[0]["meta"]["stage"] = _n
)PY";
    config.inputCount = 1;
    config.outputCount = 1;
    ConfigureF32Output(config, "[1]");

    const auto createResult = module->create("python_nested_attrs", config, inputs);
    if (createResult != Result::SUCCESS && OptionalPythonRuntimeUnavailable()) {
        SUCCEED("Skipping Python nested attribute test because the local Python runtime is unavailable.");
        return;
    }

    REQUIRE(createResult == Result::SUCCESS);

    Runtime runtime("python", DeviceType::CPU, RuntimeType::PYTHON);
    REQUIRE(runtime.create({{"python_nested_attrs", module}}) == Result::SUCCESS);

    std::unordered_set<std::string> skippedModules;
    std::unordered_set<std::string> failedModules;
    REQUIRE(runtime.compute({"python_nested_attrs"}, skippedModules, failedModules) == Result::SUCCESS);
    REQUIRE(skippedModules.empty());

    const Tensor& output = module->outputs().at("output0").tensor;
    REQUIRE(output.hasAttribute("meta"));
    {
        const auto meta = std::any_cast<Parser::Map>(output.attribute("meta"));
        REQUIRE(std::any_cast<I64>(meta.at("stage")) == 1);
    }

    REQUIRE(runtime.compute({"python_nested_attrs"}, skippedModules, failedModules) == Result::SUCCESS);
    {
        const auto meta = std::any_cast<Parser::Map>(output.attribute("meta"));
        REQUIRE(std::any_cast<I64>(meta.at("stage")) == 2);
    }

    REQUIRE(runtime.destroy() == Result::SUCCESS);
    REQUIRE(module->destroy() == Result::SUCCESS);
}

TEST_CASE("Python module creates a configured output without inputs", "[modules][python][module]") {
    std::shared_ptr<Module> module;
    REQUIRE(Registry::BuildModule("python", DeviceType::CPU, RuntimeType::PYTHON, "generic", module) ==
            Result::SUCCESS);

    Modules::Python config;
    config.code = R"PY(def compute(ctx):
    ctx.outputs[0][...] = 7.0
)PY";
    config.inputCount = 0;
    config.outputCount = 1;
    config.outputTensorSpecs = {{.shape = "[2, 3]", .dtype = "F64"}};

    const auto createResult = module->create("python_configured_source", config, {});
    if (createResult != Result::SUCCESS && OptionalPythonRuntimeUnavailable()) {
        SUCCEED("Skipping Python configured-source test because the local Python runtime is unavailable.");
        return;
    }

    REQUIRE(createResult == Result::SUCCESS);

    Runtime runtime("python", DeviceType::CPU, RuntimeType::PYTHON);
    REQUIRE(runtime.create({{"python_configured_source", module}}) == Result::SUCCESS);

    std::unordered_set<std::string> skippedModules;
    std::unordered_set<std::string> failedModules;
    REQUIRE(runtime.compute({"python_configured_source"}, skippedModules, failedModules) == Result::SUCCESS);
    REQUIRE(skippedModules.empty());

    const Tensor& output = module->outputs().at("output0").tensor;
    REQUIRE(output.dtype() == DataType::F64);
    REQUIRE(output.shape() == Shape{2, 3});
    for (Index i = 0; i < output.size(); ++i) {
        REQUIRE(std::abs(output.at<F64>(i) - 7.0) < 1e-9);
    }

    REQUIRE(runtime.destroy() == Result::SUCCESS);
    REQUIRE(module->destroy() == Result::SUCCESS);
}

TEST_CASE("Python module reports invalid source without failing flowgraph execution", "[modules][python][module]") {
    std::shared_ptr<Module> module;
    REQUIRE(Registry::BuildModule("python", DeviceType::CPU, RuntimeType::PYTHON, "generic", module) ==
            Result::SUCCESS);

    Tensor input;
    REQUIRE(input.create(DeviceType::CPU, DataType::F32, {4}) == Result::SUCCESS);

    TensorMap inputs;
    inputs["input0"].produced("source", "output", input);

    Modules::Python config;
    config.code = "def compute(ctx):\n    ctx[\"outputs\"][\n";
    config.inputCount = 1;
    config.outputCount = 1;
    ConfigureF32Output(config, "[4]");

    const auto createResult = module->create("python_invalid_source", config, inputs);
    if (createResult != Result::SUCCESS && OptionalPythonRuntimeUnavailable()) {
        SUCCEED("Skipping Python invalid-source test because the local Python runtime is unavailable.");
        return;
    }

    REQUIRE(createResult == Result::SUCCESS);

    const auto diagnostic = module->context()->runtime()->diagnostic();
    REQUIRE_FALSE(diagnostic.healthy);
    REQUIRE(diagnostic.status == "Source error.");
    REQUIRE_FALSE(diagnostic.console.empty());

    bool sawSyntaxError = false;
    bool sawLineNumber = false;
    for (const auto& line : diagnostic.console) {
        if (line.find("SyntaxError") != std::string::npos) {
            sawSyntaxError = true;
        }
        if (line.find("line 2") != std::string::npos) {
            sawLineNumber = true;
        }
    }
    REQUIRE(sawSyntaxError);
    REQUIRE(sawLineNumber);

    Runtime runtime("python", DeviceType::CPU, RuntimeType::PYTHON);
    REQUIRE(runtime.create({{"python_invalid_source", module}}) == Result::SUCCESS);

    std::unordered_set<std::string> skippedModules;
    std::unordered_set<std::string> failedModules;
    REQUIRE(runtime.compute({"python_invalid_source"}, skippedModules, failedModules) == Result::SUCCESS);
    REQUIRE(skippedModules.contains("python_invalid_source"));

    REQUIRE(runtime.destroy() == Result::SUCCESS);
    REQUIRE(module->destroy() == Result::SUCCESS);
}

TEST_CASE("Python module reports runtime exceptions without failing flowgraph execution", "[modules][python][module]") {
    std::shared_ptr<Module> module;
    REQUIRE(Registry::BuildModule("python", DeviceType::CPU, RuntimeType::PYTHON, "generic", module) ==
            Result::SUCCESS);

    Tensor input;
    REQUIRE(input.create(DeviceType::CPU, DataType::F32, {4}) == Result::SUCCESS);

    TensorMap inputs;
    inputs["input0"].produced("source", "output", input);

    Modules::Python config;
    config.code = R"PY(def compute(ctx):
    print("before boom")
    raise RuntimeError("boom")
)PY";
    config.inputCount = 1;
    config.outputCount = 1;
    ConfigureF32Output(config, "[4]");

    const auto createResult = module->create("python_runtime_error", config, inputs);
    if (createResult != Result::SUCCESS && OptionalPythonRuntimeUnavailable()) {
        SUCCEED("Skipping Python runtime-error test because the local Python runtime is unavailable.");
        return;
    }

    REQUIRE(createResult == Result::SUCCESS);

    Runtime runtime("python", DeviceType::CPU, RuntimeType::PYTHON);
    REQUIRE(runtime.create({{"python_runtime_error", module}}) == Result::SUCCESS);

    std::unordered_set<std::string> skippedModules;
    std::unordered_set<std::string> failedModules;
    REQUIRE(runtime.compute({"python_runtime_error"}, skippedModules, failedModules) == Result::SUCCESS);
    REQUIRE(skippedModules.contains("python_runtime_error"));

    const auto diagnostic = module->context()->runtime()->diagnostic();
    REQUIRE_FALSE(diagnostic.healthy);
    REQUIRE(diagnostic.status == "Compute error.");
    REQUIRE_FALSE(diagnostic.console.empty());
    int printCount = 0;
    bool sawError = false;
    for (const auto& line : diagnostic.console) {
        printCount += line.find("before boom") != std::string::npos ? 1 : 0;
        sawError = sawError || line.find("RuntimeError: boom") != std::string::npos;
    }
    REQUIRE(printCount == 1);
    REQUIRE(sawError);

    skippedModules.clear();
    failedModules.clear();
    REQUIRE(runtime.compute({"python_runtime_error"}, skippedModules, failedModules) == Result::SUCCESS);
    REQUIRE(skippedModules.contains("python_runtime_error"));

    const auto retryDiagnostic = module->context()->runtime()->diagnostic();
    int retryPrintCount = 0;
    for (const auto& line : retryDiagnostic.console) {
        retryPrintCount += line.find("before boom") != std::string::npos ? 1 : 0;
    }
    REQUIRE(retryPrintCount == 1);

    REQUIRE(runtime.destroy() == Result::SUCCESS);
    REQUIRE(module->destroy() == Result::SUCCESS);
}

TEST_CASE("Python module captures stdout for the active module", "[modules][python][module]") {
    std::shared_ptr<Module> firstModule;
    std::shared_ptr<Module> secondModule;
    REQUIRE(Registry::BuildModule("python", DeviceType::CPU, RuntimeType::PYTHON, "generic", firstModule) ==
            Result::SUCCESS);
    REQUIRE(Registry::BuildModule("python", DeviceType::CPU, RuntimeType::PYTHON, "generic", secondModule) ==
            Result::SUCCESS);

    Tensor input;
    REQUIRE(input.create(DeviceType::CPU, DataType::F32, {1}) == Result::SUCCESS);
    input.at<F32>(0) = 1.0f;

    TensorMap inputs;
    inputs["input0"].produced("source", "output", input);

    Modules::Python firstConfig;
    firstConfig.code = R"PY(def compute(ctx):
    print("first module")
    ctx.outputs[0][...] = ctx.inputs[0]
)PY";
    firstConfig.inputCount = 1;
    firstConfig.outputCount = 1;
    ConfigureF32Output(firstConfig, "[1]");

    const auto firstCreateResult = firstModule->create("python_stdout_first", firstConfig, inputs);
    if (firstCreateResult != Result::SUCCESS && OptionalPythonRuntimeUnavailable()) {
        SUCCEED("Skipping Python stdout attribution test because the local Python runtime is unavailable.");
        return;
    }
    REQUIRE(firstCreateResult == Result::SUCCESS);

    Modules::Python secondConfig;
    secondConfig.code = R"PY(def compute(ctx):
    print("second module")
    ctx.outputs[0][...] = ctx.inputs[0]
)PY";
    secondConfig.inputCount = 1;
    secondConfig.outputCount = 1;
    ConfigureF32Output(secondConfig, "[1]");
    REQUIRE(secondModule->create("python_stdout_second", secondConfig, inputs) == Result::SUCCESS);

    Runtime runtime("python", DeviceType::CPU, RuntimeType::PYTHON);
    REQUIRE(runtime.create({{"python_stdout_first", firstModule}, {"python_stdout_second", secondModule}}) ==
            Result::SUCCESS);

    std::unordered_set<std::string> skippedModules;
    std::unordered_set<std::string> failedModules;
    REQUIRE(runtime.compute({"python_stdout_first"}, skippedModules, failedModules) == Result::SUCCESS);
    REQUIRE(skippedModules.empty());

    const auto firstDiagnostic = firstModule->context()->runtime()->diagnostic();
    const auto secondDiagnostic = secondModule->context()->runtime()->diagnostic();

    bool firstSawFirst = false;
    for (const auto& line : firstDiagnostic.console) {
        firstSawFirst = firstSawFirst || line.find("first module") != std::string::npos;
    }

    bool secondSawFirst = false;
    for (const auto& line : secondDiagnostic.console) {
        secondSawFirst = secondSawFirst || line.find("first module") != std::string::npos;
    }

    REQUIRE(firstSawFirst);
    REQUIRE_FALSE(secondSawFirst);

    REQUIRE(runtime.destroy() == Result::SUCCESS);
    REQUIRE(firstModule->destroy() == Result::SUCCESS);
    REQUIRE(secondModule->destroy() == Result::SUCCESS);
}

TEST_CASE("Python module rejects malformed output tensor specs", "[modules][python][module]") {
    const std::vector<Modules::Python::TensorSpec> invalidSpecs = {
        {.shape = "[1x]"},
        {.shape = "[-1]"},
        {.shape = "[0]"},
        {.dtype = "BAD"},
        {.device = "metal"},
        {.device = ""},
    };

    for (const auto& spec : invalidSpecs) {
        std::shared_ptr<Module> module;
        REQUIRE(Registry::BuildModule("python", DeviceType::CPU, RuntimeType::PYTHON, "generic", module) ==
                Result::SUCCESS);

        Modules::Python config;
        config.code = "def compute(ctx):\n    pass\n";
        config.inputCount = 0;
        config.outputCount = 1;
        config.outputTensorSpecs = {spec};

        REQUIRE(module->create("python_bad_tensor_spec", config, {}) == Result::ERROR);
    }
}
