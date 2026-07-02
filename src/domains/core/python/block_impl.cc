#include <jetstream/domains/core/python/block.hh>
#include <jetstream/detail/block_impl.hh>

#include <jetstream/domains/core/python/module.hh>
#include <jetstream/module_context.hh>
#include <jetstream/runtime_context.hh>

#include <algorithm>
#include <any>
#include <cctype>
#include <exception>
#include <sstream>

namespace Jetstream::Blocks {

namespace {

constexpr U64 kMaxPythonPorts = 64;

std::string Trim(std::string value) {
    while (!value.empty() && std::isspace(static_cast<unsigned char>(value.front()))) {
        value.erase(value.begin());
    }
    while (!value.empty() && std::isspace(static_cast<unsigned char>(value.back()))) {
        value.pop_back();
    }
    return value;
}

std::string ToLower(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](const unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
    return value;
}

std::string ToUpper(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](const unsigned char ch) {
        return static_cast<char>(std::toupper(ch));
    });
    return value;
}

bool IsUnsignedInteger(const std::string& value) {
    return !value.empty() &&
           std::all_of(value.begin(), value.end(), [](const unsigned char ch) {
               return std::isdigit(ch);
           });
}

bool PythonDataTypeSupported(const DataType dtype) {
    switch (dtype) {
        case DataType::I8:
        case DataType::I16:
        case DataType::I32:
        case DataType::I64:
        case DataType::U8:
        case DataType::U16:
        case DataType::U32:
        case DataType::U64:
        case DataType::F32:
        case DataType::F64:
        case DataType::CF32:
        case DataType::CF64:
            return true;
        default:
            return false;
    }
}

template<typename Config>
void NormalizeOutputSpecs(Config& config) {
    config.outputTensorSpecs.resize(config.outputCount);
}

Result ValidateDataTypeSpec(const std::string& spec, const std::string& label) {
    const auto dtype = NameToDataType(ToUpper(Trim(spec)));
    if (dtype == DataType::None || !PythonDataTypeSupported(dtype)) {
        JST_ERROR("[PYTHON] Invalid {} data type '{}'.", label, spec);
        return Result::ERROR;
    }

    return Result::SUCCESS;
}

Result ValidateDeviceSpec(const std::string& spec, const std::string& label) {
    const auto normalized = ToLower(Trim(spec));
    if (normalized.empty()) {
        JST_ERROR("[PYTHON] {} device cannot be empty.", label);
        return Result::ERROR;
    }

    const auto device = StringToDevice(normalized);
    if (device == DeviceType::None) {
        JST_ERROR("[PYTHON] Invalid {} device '{}'.", label, spec);
        return Result::ERROR;
    }

    if (device != DeviceType::CPU) {
        JST_ERROR("[PYTHON] Python tensor {} device must be CPU today (got {}).", label, device);
        return Result::ERROR;
    }

    return Result::SUCCESS;
}

Result ValidateShapeSpec(const std::string& spec, const std::string& label) {
    auto normalized = Trim(spec);
    if (normalized.empty()) {
        JST_ERROR("[PYTHON] {} shape cannot be empty.", label);
        return Result::ERROR;
    }

    if (normalized.front() == '[') {
        if (normalized.back() != ']') {
            JST_ERROR("[PYTHON] Invalid {} shape '{}'.", label, spec);
            return Result::ERROR;
        }
        normalized = Trim(normalized.substr(1, normalized.size() - 2));
    } else if (normalized.back() == ']') {
        JST_ERROR("[PYTHON] Invalid {} shape '{}'.", label, spec);
        return Result::ERROR;
    }

    if (normalized.empty()) {
        JST_ERROR("[PYTHON] {} shape cannot be empty.", label);
        return Result::ERROR;
    }

    std::stringstream stream(normalized);
    std::string token;
    while (std::getline(stream, token, ',')) {
        token = Trim(token);
        if (token.empty()) {
            JST_ERROR("[PYTHON] Invalid {} shape '{}'.", label, spec);
            return Result::ERROR;
        }

        if (!IsUnsignedInteger(token)) {
            JST_ERROR("[PYTHON] Invalid {} shape dimension '{}'.", label, token);
            return Result::ERROR;
        }

        try {
            if (std::stoull(token) == 0) {
                JST_ERROR("[PYTHON] {} shape dimensions must be greater than zero.", label);
                return Result::ERROR;
            }
        } catch (const std::exception&) {
            JST_ERROR("[PYTHON] Invalid {} shape dimension '{}'.", label, token);
            return Result::ERROR;
        }
    }

    return Result::SUCCESS;
}

template<typename Config>
Result ValidatePortSpecs(const Config& config) {
    for (U64 i = 0; i < config.outputCount; ++i) {
        const auto label = "output" + std::to_string(i);
        const auto& spec = config.outputTensorSpecs.at(i);
        JST_CHECK(ValidateShapeSpec(spec.shape, label));
        JST_CHECK(ValidateDataTypeSpec(spec.dtype, label));
        JST_CHECK(ValidateDeviceSpec(spec.device, label));
    }

    return Result::SUCCESS;
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

    if (config.code.empty()) {
        JST_ERROR("[PYTHON] Code cannot be empty.");
        return Result::ERROR;
    }

    if (config.inputCount > kMaxPythonPorts || config.outputCount > kMaxPythonPorts) {
        JST_ERROR("[PYTHON] Input and output counts must be at most {}.", kMaxPythonPorts);
        return Result::ERROR;
    }

    NormalizeOutputSpecs(config);

    JST_CHECK(ValidatePortSpecs(config));

    if (inputCount != config.inputCount || outputCount != config.outputCount) {
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

    return Result::SUCCESS;
}

Result PythonImpl::define() {
    for (U64 i = 0; i < inputCount; ++i) {
        const auto index = std::to_string(i);
        JST_CHECK(defineInterfaceInput(InputPortName(i),
                                       "Input " + index,
                                       "Tensor exposed as ctx.inputs[" + index + "]."));
    }

    for (U64 i = 0; i < outputCount; ++i) {
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
                                    "int:"));
    JST_CHECK(defineInterfaceConfig("outputCount",
                                    "Output Count",
                                    "Number of output tensor ports.",
                                    "int:"));
    for (U64 i = 0; i < outputCount; ++i) {
        const auto index = std::to_string(i);
        JST_CHECK(defineInterfaceConfig("outputTensor" + index,
                                        "Output " + index,
                                        "Tensor shape, data type, and device for output " + index + ".",
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

JST_REGISTER_BLOCK(PythonImpl);

}  // namespace Jetstream::Blocks
