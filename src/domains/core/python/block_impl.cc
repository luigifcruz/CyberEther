#include <jetstream/domains/core/python/block.hh>
#include <jetstream/detail/block_impl.hh>

#include <jetstream/domains/core/python/module.hh>
#include <jetstream/module_context.hh>
#include <jetstream/runtime_context.hh>

#include <algorithm>
#include <any>

namespace Jetstream::Blocks {

namespace {

constexpr U64 kMaxPythonPorts = 64;

template<typename Config>
void NormalizeOutputSpecs(Config& config) {
    config.outputTensorSpecs.resize(config.outputCount);
}

std::string InputPortName(const U64 index) {
    return "input" + std::to_string(index);
}

std::string OutputPortName(const U64 index) {
    return "output" + std::to_string(index);
}

}  // namespace

struct PythonImpl : public Block::Impl, public DynamicConfig<Blocks::Python> {
    Result validate() override;
    Result configure() override;
    Result define() override;
    Result create() override;

 protected:
    std::shared_ptr<Modules::Python> moduleConfig = std::make_shared<Modules::Python>();
};

Result PythonImpl::validate() {
    auto config = *candidate();

    if (runtime() != RuntimeType::PYTHON) {
        JST_ERROR("[PYTHON] Block must be created with the Python runtime.");
        return Result::ERROR;
    }

    if (config.inputCount > kMaxPythonPorts || config.outputCount > kMaxPythonPorts) {
        JST_ERROR("[PYTHON] Input and output counts must be at most {}.", kMaxPythonPorts);
        return Result::ERROR;
    }

    if (inputCount != config.inputCount ||
        outputCount != config.outputCount ||
        throttled != config.throttled) {
        return Result::RECREATE;
    }

    return Result::SUCCESS;
}

Result PythonImpl::configure() {
    NormalizeOutputSpecs(*this);

    moduleConfig->code = code;
    moduleConfig->inputCount = inputCount;
    moduleConfig->outputCount = outputCount;
    moduleConfig->outputTensorSpecs = outputTensorSpecs;
    moduleConfig->throttled = throttled;

    return Result::SUCCESS;
}

Result PythonImpl::define() {
    const auto& config = *candidate();
    const U64 interfaceInputCount = std::min(config.inputCount, kMaxPythonPorts);
    const U64 interfaceOutputCount = std::min(config.outputCount, kMaxPythonPorts);

    for (U64 i = 0; i < interfaceInputCount; ++i) {
        const auto index = std::to_string(i);
        JST_CHECK(defineInterfaceInput(InputPortName(i),
                                       "Input " + index,
                                       "Tensor exposed as ctx.inputs[" + index + "]."));
    }

    for (U64 i = 0; i < interfaceOutputCount; ++i) {
        const auto index = std::to_string(i);
        JST_CHECK(defineInterfaceOutput(OutputPortName(i),
                                        "Output " + index,
                                        "Tensor exposed as ctx.outputs[" + index + "]."));
    }

    JST_CHECK(defineInterfaceConfig("code",
                                    "Code",
                                    "Python source defining compute(ctx).",
                                    "python"));
    JST_CHECK(defineInterfaceConfig("inputCount",
                                    "Input Count",
                                    "Number of input tensor ports.",
                                    "uint:"));
    JST_CHECK(defineInterfaceConfig("outputCount",
                                    "Output Count",
                                    "Number of output tensor ports.",
                                    "uint:"));
    JST_CHECK(defineInterfaceConfig("throttled",
                                    "Throttled",
                                    "Run compute at a slow fixed rate instead of every cycle.",
                                    "bool"));
    for (U64 i = 0; i < interfaceOutputCount; ++i) {
        const auto index = std::to_string(i);
        JST_CHECK(defineInterfaceConfig("outputTensor" + index,
                                        "Output " + index,
                                        "Tensor shape, data type, device, and signal axes for output " + index + ".",
                                        "tensor-config:" + index));
    }
    JST_CHECK(defineInterfaceMetric("pythonDiagnostic",
                                    "Python Diagnostic",
                                    "Console output from the Python runtime context.",
                                    "private-python-diagnostic",
                                    [this]() -> std::any {
        const auto module = moduleHandle("python");
        if (!module || !module->context() || !module->context()->runtime()) {
            return Runtime::Context::Diagnostic{};
        }

        return module->context()->runtime()->diagnostic();
    }));

    return Result::SUCCESS;
}

Result PythonImpl::create() {
    TensorMap moduleInputs;
    for (U64 i = 0; i < inputCount; ++i) {
        const auto port = InputPortName(i);
        moduleInputs[port] = inputs().at(port);
    }

    JST_CHECK(moduleCreate("python", moduleConfig, moduleInputs));

    for (U64 i = 0; i < outputCount; ++i) {
        const auto port = OutputPortName(i);
        JST_CHECK(moduleExposeOutput(port, {"python", port}));
    }

    return Result::SUCCESS;
}

JST_REGISTER_BLOCK(PythonImpl, {"python"});

}  // namespace Jetstream::Blocks
