#include "module_impl.hh"

#include <algorithm>
#include <cctype>
#include <charconv>
#include <sstream>

#include <jetstream/memory/axis.hh>
#include <jetstream/tools/numeric.hh>

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

        Index dimension = 0;
        const auto* end = token.data() + token.size();
        const auto [position, error] = std::from_chars(token.data(), end, dimension);
        if (error != std::errc{} || position != end) {
            JST_ERROR("[PYTHON] Invalid {} shape dimension '{}'.", label, token);
            return Result::ERROR;
        }

        if (dimension == 0) {
            JST_ERROR("[PYTHON] {} shape dimensions must be greater than zero.", label);
            return Result::ERROR;
        }
        parsed.push_back(dimension);
    }

    shape = std::move(parsed);
    return Result::SUCCESS;
}

Result ParseSignalAxesSpec(const std::string& spec,
                           const std::string& label,
                           const Shape& shape,
                           std::map<std::string, std::any>& attributes) {
    auto normalized = Trim(spec);
    if (normalized.empty()) {
        return Result::SUCCESS;
    }

    if (normalized.front() != '[' || normalized.back() != ']') {
        JST_ERROR("[PYTHON] Signal axes '{}' of {} must use the [B, C, S] form.", spec, label);
        return Result::ERROR;
    }
    normalized = Trim(normalized.substr(1, normalized.size() - 2));
    if (normalized.empty()) {
        JST_ERROR("[PYTHON] Signal axes of {} cannot be empty.", label);
        return Result::ERROR;
    }
    if (normalized.back() == ',') {
        JST_ERROR("[PYTHON] Signal axes of {} contain an empty entry.", label);
        return Result::ERROR;
    }

    std::vector<std::pair<char, Index>> roles;
    std::stringstream stream(normalized);
    std::string token;
    Index axis = 0;
    while (std::getline(stream, token, ',')) {
        token = Trim(token);
        if (token.empty()) {
            JST_ERROR("[PYTHON] Signal axes of {} contain an empty entry.", label);
            return Result::ERROR;
        }
        if (token.size() != 1 ||
            (token[0] != 'B' && token[0] != 'C' && token[0] != 'S' && token[0] != '_')) {
            JST_ERROR("[PYTHON] Signal axes entry '{}' of {} must be one of B, C, S, or _.", token, label);
            return Result::ERROR;
        }
        if (token[0] != '_') {
            for (const auto& [role, _] : roles) {
                if (role == token[0]) {
                    JST_ERROR("[PYTHON] Signal axes of {} use role '{}' more than once.", label, token[0]);
                    return Result::ERROR;
                }
            }
        }
        roles.emplace_back(token[0], axis++);
    }

    bool hasRole = false;
    for (const auto& [role, _] : roles) {
        hasRole = hasRole || role != '_';
    }
    if (!hasRole) {
        JST_ERROR("[PYTHON] Signal axes of {} must declare at least one of B, C, or S.", label);
        return Result::ERROR;
    }
    if (axis > shape.size()) {
        JST_ERROR("[PYTHON] Signal axes of {} describe {} axes but shape has rank {}.",
                  label, axis, shape.size());
        return Result::ERROR;
    }

    for (const auto& [role, index] : roles) {
        switch (role) {
            case 'B':
                attributes[std::string(BatchAxisAttribute)] = Index{index};
                break;
            case 'C':
                attributes[std::string(ChannelAxisAttribute)] = Index{index};
                break;
            case 'S':
                attributes[std::string(SampleAxisAttribute)] = Index{index};
                break;
        }
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

    std::vector<OutputPlan> outputPlan;
    outputPlan.reserve(config.outputCount);
    for (U64 i = 0; i < config.outputCount; ++i) {
        const auto label = "output" + std::to_string(i);
        const auto& spec = config.outputTensorSpecs.at(i);
        OutputPlan output;

        JST_CHECK(ParseDataTypeSpec(spec.dtype, label, output.dtype));
        JST_CHECK(ParseShapeSpec(spec.shape, label, output.shape));
        JST_CHECK(ParseDeviceSpec(spec.device, label, output.device));
        JST_CHECK(ParseSignalAxesSpec(spec.axes, label, output.shape, output.attributes));

        U64 elementCount = 1;
        for (const U64 dimension : output.shape) {
            if (!detail::CheckedMultiply(elementCount, dimension, elementCount)) {
                JST_ERROR("[PYTHON] {} shape exceeds the supported layout range.", label);
                return Result::ERROR;
            }
        }

        U64 sizeBytes = 0;
        if (!detail::CheckedMultiply(elementCount,
                                     static_cast<U64>(DataTypeSize(output.dtype)),
                                     sizeBytes)) {
            JST_ERROR("[PYTHON] {} exceeds the supported byte range.", label);
            return Result::ERROR;
        }

        output.elementCount = elementCount;
        output.sizeBytes = sizeBytes;
        outputPlan.push_back(std::move(output));
    }

    candidateOutputPlan = std::move(outputPlan);
    return Result::SUCCESS;
}

Result PythonImpl::define() {
    JST_CHECK(defineTaint(Module::Taint::DISCONTIGUOUS));
    JST_CHECK(defineTaint(Module::Taint::CROSS_DEVICE));

    if (throttled) {
        JST_CHECK(defineTaint(Module::Taint::THROTTLED));
    }

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
    for (U64 i = 0; i < candidateOutputPlan.size(); ++i) {
        const auto& plan = candidateOutputPlan[i];
        Tensor output;
        JST_CHECK(output.create(plan.device, plan.dtype, plan.shape));

        for (const auto& [key, value] : plan.attributes) {
            JST_CHECK(output.setAttribute(key, value));
        }

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
