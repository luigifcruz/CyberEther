#include "jetstream/parser.hh"
#include "jetstream/memory/types.hh"

#include "jetstream/runtime.hh"
#include "jetstream/scheduler.hh"

namespace Jetstream {

//
// Helper Functions
//

template<typename T>
static T StringToComplex(const std::string& s) {
    using ST = typename NumericTypeInfo<T>::subtype;

    ST real = 0.0;
    ST imag = 0.0;
    char op = '+';

    std::stringstream ss(s);
    ss >> real;      // Extract real part
    ss >> op;        // Extract '+' or '-'
    ss >> imag;      // Extract imaginary part

    if (op == '-') {
        imag = -imag;
    }

    return T(real, imag);
}

std::vector<std::string> Parser::SplitString(const std::string& str, const std::string& delimiter) {
    std::vector<std::string> result;
    size_t pos = 0;
    size_t lastPos = 0;
    while ((pos = str.find(delimiter, lastPos)) != std::string::npos) {
        result.push_back(str.substr(lastPos, pos - lastPos));
        lastPos = pos + delimiter.length();
    }
    result.push_back(str.substr(lastPos));
    return result;
}

//
// StringToTyped Specializations
//

template<>
Result Parser::StringToTyped<Tensor>(const std::string& encoded, Tensor& variable) {
    JST_TRACE("Deserializing: Trying to convert 'std::any' into 'Tensor'.");
    variable = std::any_cast<Tensor>(encoded);
    return Result::SUCCESS;
}

template<>
Result Parser::StringToTyped<std::string>(const std::string& encoded, std::string& variable) {
    JST_TRACE("Deserializing: Trying to convert 'std::any' into 'std::string'.");
    variable = std::any_cast<std::string>(encoded);
    return Result::SUCCESS;
}

template<>
Result Parser::StringToTyped<I32>(const std::string& encoded, I32& variable) {
    JST_TRACE("Deserializing: Trying to convert 'std::any' into 'I32'.");
    variable = std::stoi(std::any_cast<std::string>(encoded));
    return Result::SUCCESS;
}

template<>
Result Parser::StringToTyped<U64>(const std::string& encoded, U64& variable) {
    JST_TRACE("Deserializing: Trying to convert 'std::any' into 'U64'.");
    variable = std::stoull(std::any_cast<std::string>(encoded));
    return Result::SUCCESS;
}

template<>
Result Parser::StringToTyped<F32>(const std::string& encoded, F32& variable) {
    JST_TRACE("Deserializing: Trying to convert 'std::any' into 'F32'.");
    variable = std::stof(std::any_cast<std::string>(encoded));
    return Result::SUCCESS;
}

template<>
Result Parser::StringToTyped<F64>(const std::string& encoded, F64& variable) {
    JST_TRACE("Deserializing: Trying to convert 'std::any' into 'F64'.");
    variable = std::stod(std::any_cast<std::string>(encoded));
    return Result::SUCCESS;
}

template<>
Result Parser::StringToTyped<CF32>(const std::string& encoded, CF32& variable) {
    JST_TRACE("Deserializing: Trying to convert 'std::any' into 'CF32'.");
    variable = StringToComplex<CF32>(std::any_cast<std::string>(encoded));
    return Result::SUCCESS;
}

template<>
Result Parser::StringToTyped<bool>(const std::string& encoded, bool& variable) {
    JST_TRACE("Deserializing: Trying to convert 'std::any' into 'bool'.");
    std::string lower_s = std::any_cast<std::string>(encoded);
    std::transform(lower_s.begin(), lower_s.end(), lower_s.begin(), ::tolower);
    variable = (lower_s == "true" || lower_s == "1");
    return Result::SUCCESS;
}

template<>
Result Parser::StringToTyped<DeviceType>(const std::string& encoded, DeviceType& variable) {
    JST_TRACE("Deserializing: Trying to convert 'std::any' into 'DeviceType'.");
    variable = StringToDevice(encoded);
    return Result::SUCCESS;
}

template<>
Result Parser::StringToTyped<RuntimeType>(const std::string& encoded, RuntimeType& variable) {
    JST_TRACE("Deserializing: Trying to convert 'std::any' into 'RuntimeType'.");
    variable = StringToRuntime(encoded);
    return Result::SUCCESS;
}

template<>
Result Parser::StringToTyped<SchedulerType>(const std::string& encoded, SchedulerType& variable) {
    JST_TRACE("Deserializing: Trying to convert 'std::any' into 'SchedulerType'.");
    variable = StringToScheduler(encoded);
    return Result::SUCCESS;
}

template<>
Result Parser::StringToTyped<std::vector<U64>>(const std::string& encoded, std::vector<U64>& variable) {
    JST_TRACE("Deserializing: Trying to convert 'std::any' into 'std::vector<U64>'.");
    std::string str = encoded;
    std::erase(str, '[');
    std::erase(str, ']');
    const auto& values = SplitString(str, ", ");
    variable = std::vector<U64>(values.size());
    std::transform(values.begin(), values.end(), variable.begin(), [](const std::string& s){ return std::stoull(s); });
    return Result::SUCCESS;
}

template<>
Result Parser::StringToTyped<std::vector<F64>>(const std::string& encoded, std::vector<F64>& variable) {
    JST_TRACE("Deserializing: Trying to convert 'std::any' into 'std::vector<F64>'.");
    std::string str = encoded;
    std::erase(str, '[');
    std::erase(str, ']');
    const auto& values = SplitString(str, ", ");
    variable = std::vector<F64>(values.size());
    std::transform(values.begin(), values.end(), variable.begin(), [](const std::string& s){ return std::stod(s); });
    return Result::SUCCESS;
}

template<>
Result Parser::StringToTyped<std::vector<F32>>(const std::string& encoded, std::vector<F32>& variable) {
    JST_TRACE("Deserializing: Trying to convert 'std::any' into 'std::vector<F32>'.");
    std::string str = encoded;
    std::erase(str, '[');
    std::erase(str, ']');
    const auto& values = SplitString(str, ", ");
    variable = std::vector<F32>(values.size());
    std::transform(values.begin(), values.end(), variable.begin(), [](const std::string& s){ return std::stof(s); });
    return Result::SUCCESS;
}

template<>
Result Parser::StringToTyped<Range<F32>>(const std::string& encoded, Range<F32>& variable) {
    JST_TRACE("Deserializing: Trying to convert 'std::any' into 'Range<F32>'.");
    const auto& values = SplitString(std::any_cast<std::string>(encoded), ", ");
    JST_ASSERT(values.size() == 2, "Unexpected number of values.");
    variable = Range<F32>{std::stof(values[0]), std::stof(values[1])};
    return Result::SUCCESS;
}

template<>
Result Parser::StringToTyped<Extent2D<U64>>(const std::string& encoded, Extent2D<U64>& variable) {
    JST_TRACE("Deserializing: Trying to convert 'std::any' into 'Extent2D<U64>'.");
    const auto& values = SplitString(std::any_cast<std::string>(encoded), ", ");
    JST_ASSERT(values.size() == 2, "Unexpected number of values.");
    variable = Extent2D<U64>{std::stoull(values[0]), std::stoull(values[1])};
    return Result::SUCCESS;
}

template<>
Result Parser::StringToTyped<Extent2D<F32>>(const std::string& encoded, Extent2D<F32>& variable) {
    JST_TRACE("Deserializing: Trying to convert 'std::any' into 'Extent2D<F32>'.");
    const auto& values = SplitString(std::any_cast<std::string>(encoded), ", ");
    JST_ASSERT(values.size() == 2, "Unexpected number of values.");
    variable = Extent2D<F32>{std::stof(values[0]), std::stof(values[1])};
    return Result::SUCCESS;
}

//
// StringToTyped Specializations
//

Result Parser::TypedToString(const std::any& variable, std::string& encoded) {
    if (variable.type() == typeid(std::string)) {
        const auto& string_value = std::any_cast<std::string>(variable);
        encoded = jst::fmt::format("{}", string_value);
        return Result::SUCCESS;
    }

    if (variable.type() == typeid(I32)) {
        encoded = jst::fmt::format("{}", std::any_cast<I32>(variable));
        return Result::SUCCESS;
    }

    if (variable.type() == typeid(U64)) {
        encoded = jst::fmt::format("{}", std::any_cast<U64>(variable));
        return Result::SUCCESS;
    }

    if (variable.type() == typeid(F32)) {
        encoded = jst::fmt::format("{}", std::any_cast<F32>(variable));
        return Result::SUCCESS;
    }

    if (variable.type() == typeid(F64)) {
        encoded = jst::fmt::format("{}", std::any_cast<F64>(variable));
        return Result::SUCCESS;
    }

    if (variable.type() == typeid(CF32)) {
        const auto& complex = std::any_cast<CF32>(variable);
        encoded = jst::fmt::format("{}{}{}", complex.real(), complex.imag() < 0 ? "-" : "+", complex.imag());
        return Result::SUCCESS;
    }

    if (variable.type() == typeid(bool)) {
        encoded = jst::fmt::format("{}", std::any_cast<bool>(variable));
        return Result::SUCCESS;
    }

    if (variable.type() == typeid(std::vector<U64>)) {
        const auto& values = std::any_cast<std::vector<U64>>(variable);
        encoded = jst::fmt::format("[{}]", jst::fmt::join(values, ", "));
        return Result::SUCCESS;
    }

    if (variable.type() == typeid(std::vector<F32>)) {
        const auto& values = std::any_cast<std::vector<F32>>(variable);
        encoded = jst::fmt::format("[{}]", jst::fmt::join(values, ", "));
        return Result::SUCCESS;
    }

    if (variable.type() == typeid(std::vector<F64>)) {
        const auto& values = std::any_cast<std::vector<F64>>(variable);
        encoded = jst::fmt::format("[{}]", jst::fmt::join(values, ", "));
        return Result::SUCCESS;
    }

    if (variable.type() == typeid(Range<F32>)) {
        const auto& range = std::any_cast<Range<F32>>(variable);
        encoded = jst::fmt::format("[{}, {}]", range.min, range.max);
        return Result::SUCCESS;
    }

    if (variable.type() == typeid(Extent2D<U64>)) {
        const auto& size = std::any_cast<Extent2D<U64>>(variable);
        encoded = jst::fmt::format("[{}, {}]", size.x, size.y);
        return Result::SUCCESS;
    }

    if (variable.type() == typeid(Extent2D<F32>)) {
        const auto& size = std::any_cast<Extent2D<F32>>(variable);
        encoded = jst::fmt::format("[{}, {}]", size.x, size.y);
        return Result::SUCCESS;
    }

    JST_ERROR("[PARSER] Failed to serialize variable. Check if the input and output are compatible.");
    JST_TRACE("[PARSER] Variable type: {}", variable.type().name());
    return Result::ERROR;
}

}  // namespace Jetstream
