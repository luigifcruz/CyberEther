#include "module_impl.hh"

#include <algorithm>
#include <cctype>
#include <exception>
#include <sstream>

namespace Jetstream::Modules {

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

Module::Interface::EntryList PortOrder(const U64 count, std::string (*portName)(U64)) {
    Module::Interface::EntryList order;
    order.reserve(count);
    for (U64 i = 0; i < count; ++i) {
        order.push_back(portName(i));
    }
    return order;
}

Result ParseDataTypeSpec(const std::string& spec,
                         const std::string& label,
                         DataType& dtype) {
    const auto normalized = ToUpper(Trim(spec));
    if (normalized.empty()) {
        JST_ERROR("[PYTHON] {} data type cannot be empty.", label);
        return Result::ERROR;
    }

    dtype = NameToDataType(normalized);
    if (dtype == DataType::None || !PythonDataTypeSupported(dtype)) {
        JST_ERROR("[PYTHON] Invalid {} data type '{}'.", label, spec);
        return Result::ERROR;
    }

    return Result::SUCCESS;
}

Result ParseDeviceSpec(const std::string& spec,
                       const std::string& label,
                       DeviceType& device) {
    const auto normalized = ToLower(Trim(spec));
    if (normalized.empty()) {
        JST_ERROR("[PYTHON] {} device cannot be empty.", label);
        return Result::ERROR;
    }

    device = StringToDevice(normalized);
    if (device == DeviceType::None) {
        JST_ERROR("[PYTHON] Invalid {} device '{}'.", label, spec);
        return Result::ERROR;
    }

    return Result::SUCCESS;
}

Result ParseShapeSpec(const std::string& spec,
                      const std::string& label,
                      Shape& shape) {
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

    Shape parsed;
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
            const auto dimension = static_cast<Index>(std::stoull(token));
            if (dimension == 0) {
                JST_ERROR("[PYTHON] {} shape dimensions must be greater than zero.", label);
                return Result::ERROR;
            }
            parsed.push_back(dimension);
        } catch (const std::exception&) {
            JST_ERROR("[PYTHON] Invalid {} shape dimension '{}'.", label, token);
            return Result::ERROR;
        }
    }

    shape = std::move(parsed);
    return Result::SUCCESS;
}

Result ValidatePortSpecs(const Python& config) {
    for (U64 i = 0; i < config.outputCount; ++i) {
        const auto label = "output" + std::to_string(i);
        const auto& spec = config.outputTensorSpecs.at(i);

        DataType dtype = DataType::None;
        JST_CHECK(ParseDataTypeSpec(spec.dtype, label, dtype));

        Shape shape;
        JST_CHECK(ParseShapeSpec(spec.shape, label, shape));

        DeviceType device = DeviceType::None;
        JST_CHECK(ParseDeviceSpec(spec.device, label, device));
    }

    return Result::SUCCESS;
}

Result ResolveOutputSpec(const U64 index,
                         const std::vector<Python::TensorSpec>& specs,
                         DataType& dtype,
                         Shape& shape,
                         DeviceType& device) {
    const auto label = "output" + std::to_string(index);
    const auto& spec = specs.at(index);

    JST_CHECK(ParseDataTypeSpec(spec.dtype, label, dtype));

    JST_CHECK(ParseShapeSpec(spec.shape, label, shape));

    JST_CHECK(ParseDeviceSpec(spec.device, label, device));

    if (device != DeviceType::CPU) {
        JST_ERROR("[PYTHON] Python tensor {} device must be CPU today (got {}).", label, device);
        return Result::ERROR;
    }

    return Result::SUCCESS;
}

}  // namespace

std::string PythonImpl::inputPortName(const U64 index) {
    return "input" + std::to_string(index);
}

std::string PythonImpl::outputPortName(const U64 index) {
    return "output" + std::to_string(index);
}

void PythonImpl::normalizeOutputSpecs(Python& config) {
    config.outputTensorSpecs.resize(config.outputCount);
}

Module::Interface::EntryList PythonImpl::inputPortOrder() const {
    return PortOrder(inputCount, inputPortName);
}

Module::Interface::EntryList PythonImpl::outputPortOrder() const {
    return PortOrder(outputCount, outputPortName);
}

Result PythonImpl::validate() {
    auto config = *candidate();

    if (config.code.empty()) {
        JST_ERROR("[PYTHON] Code cannot be empty.");
        return Result::ERROR;
    }

    if (config.inputCount > kMaxPythonPorts || config.outputCount > kMaxPythonPorts) {
        JST_ERROR("[PYTHON] Input and output counts must be at most {}.", kMaxPythonPorts);
        return Result::ERROR;
    }

    normalizeOutputSpecs(config);

    JST_CHECK(ValidatePortSpecs(config));

    return Result::SUCCESS;
}

Result PythonImpl::define() {
    normalizeOutputSpecs(*this);

    for (U64 i = 0; i < inputCount; ++i) {
        JST_CHECK(defineInterfaceInput(inputPortName(i)));
    }

    for (U64 i = 0; i < outputCount; ++i) {
        JST_CHECK(defineInterfaceOutput(outputPortName(i)));
    }

    return Result::SUCCESS;
}

Result PythonImpl::create() {
    normalizeOutputSpecs(*this);

    if (outputCount == 0) {
        return Result::SUCCESS;
    }

    for (U64 i = 0; i < outputCount; ++i) {
        DataType outputDataType = DataType::None;
        Shape outputShape;
        DeviceType outputDevice = DeviceType::None;
        JST_CHECK(ResolveOutputSpec(i, outputTensorSpecs, outputDataType, outputShape, outputDevice));

        Tensor output;
        JST_CHECK(output.create(outputDevice, outputDataType, outputShape));
        outputs()[outputPortName(i)].produced(name(), outputPortName(i), output);
    }

    return Result::SUCCESS;
}

Result PythonImpl::destroy() {
    return Result::SUCCESS;
}

Result PythonImpl::reconfigure() {
    return Result::RECREATE;
}

}  // namespace Jetstream::Modules
