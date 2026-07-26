#include <cctype>
#include <utility>

#include "jetstream/parser.hh"
#include "jetstream/memory/types.hh"

#include "jetstream/runtime.hh"
#include "jetstream/scheduler.hh"

namespace Jetstream {

//
// Helper Functions
//

static bool StringIsNegative(const std::string& encoded) {
    for (const auto& ch : encoded) {
        if (std::isspace(static_cast<unsigned char>(ch))) {
            continue;
        }
        return ch == '-';
    }
    return false;
}

static bool StringIsFullyConsumed(const std::string& encoded, const std::size_t consumed) {
    return std::all_of(encoded.begin() + static_cast<std::ptrdiff_t>(consumed),
                       encoded.end(),
                       [](const unsigned char ch) { return std::isspace(ch); });
}

template<typename T>
static Result StringToInteger(const std::string& encoded, T& variable, const char* type) {
    if constexpr (std::is_unsigned_v<T>) {
        if (StringIsNegative(encoded)) {
            JST_ERROR("[PARSER] Value '{}' is out of range for '{}'.", encoded, type);
            return Result::ERROR;
        }
    }

    std::size_t consumed = 0;
    const auto value = [&] {
        if constexpr (std::is_unsigned_v<T>) {
            return std::stoull(encoded, &consumed);
        } else if constexpr (sizeof(T) <= sizeof(int)) {
            return std::stoi(encoded, &consumed);
        } else {
            return std::stoll(encoded, &consumed);
        }
    }();

    if (!StringIsFullyConsumed(encoded, consumed)) {
        JST_ERROR("[PARSER] Value '{}' is not a valid '{}'.", encoded, type);
        return Result::ERROR;
    }
    if (!std::in_range<T>(value)) {
        JST_ERROR("[PARSER] Value '{}' is out of range for '{}'.", encoded, type);
        return Result::ERROR;
    }

    variable = static_cast<T>(value);
    return Result::SUCCESS;
}

template<typename T>
static Result StringToFloating(const std::string& encoded, T& variable, const char* type) {
    std::size_t consumed = 0;
    const T value = [&] {
        if constexpr (std::is_same_v<T, F32>) {
            return std::stof(encoded, &consumed);
        } else {
            return std::stod(encoded, &consumed);
        }
    }();

    if (!StringIsFullyConsumed(encoded, consumed)) {
        JST_ERROR("[PARSER] Value '{}' is not a valid '{}'.", encoded, type);
        return Result::ERROR;
    }

    variable = value;
    return Result::SUCCESS;
}

template<typename T>
static Result StringToComplex(const std::string& encoded, T& variable) {
    using ST = typename NumericTypeInfo<T>::subtype;

    std::size_t realEnd = 0;
    ST real = [&] {
        if constexpr (std::is_same_v<ST, F32>) {
            return std::stof(encoded, &realEnd);
        } else {
            return std::stod(encoded, &realEnd);
        }
    }();

    const auto operatorPosition = std::find_if_not(
        encoded.begin() + static_cast<std::ptrdiff_t>(realEnd),
        encoded.end(),
        [](const unsigned char ch) { return std::isspace(ch); });
    if (operatorPosition == encoded.end() ||
        (*operatorPosition != '+' && *operatorPosition != '-')) {
        JST_ERROR("[PARSER] Value '{}' is not a valid complex number.", encoded);
        return Result::ERROR;
    }

    const std::string imaginary = encoded.substr(
        static_cast<std::size_t>(std::distance(encoded.begin(), operatorPosition)) + 1);
    const auto imaginaryValue = std::find_if_not(
        imaginary.begin(), imaginary.end(), [](const unsigned char ch) { return std::isspace(ch); });
    if (imaginaryValue == imaginary.end() || *imaginaryValue == '+' || *imaginaryValue == '-') {
        JST_ERROR("[PARSER] Value '{}' is not a valid complex number.", encoded);
        return Result::ERROR;
    }

    ST imag = 0.0;
    JST_CHECK(StringToFloating(imaginary, imag, "complex component"));
    if (*operatorPosition == '-') {
        imag = -imag;
    }

    variable = T(real, imag);
    return Result::SUCCESS;
}

static std::string NormalizeListString(const std::string& encoded) {
    std::string normalized = encoded;
    std::erase(normalized, '[');
    std::erase(normalized, ']');
    return normalized;
}

static std::vector<std::string> ParseListValues(const std::string& encoded) {
    const auto normalized = NormalizeListString(encoded);
    if (normalized.empty()) {
        return {};
    }

    return Parser::SplitString(normalized, ", ");
}

template<typename T>
static Result StringToVector(const std::string& encoded, std::vector<T>& variable) {
    const auto values = ParseListValues(encoded);
    std::vector<T> candidate;
    candidate.reserve(values.size());

    for (const auto& encodedValue : values) {
        T value{};
        JST_CHECK(Parser::StringToTyped(encodedValue, value));
        candidate.push_back(value);
    }

    variable = std::move(candidate);
    return Result::SUCCESS;
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
// StringToTyped overloads
//

Result Parser::StringToTypedValue(const std::string& encoded, Tensor& variable) {
    JST_TRACE("Deserializing: Trying to convert 'std::any' into 'Tensor'.");
    (void)encoded;
    (void)variable;

    JST_ERROR("[PARSER] Tensor values cannot be deserialized from a string.");
    return Result::ERROR;
}

Result Parser::StringToTypedValue(const std::string& encoded, std::string& variable) {
    JST_TRACE("Deserializing: Trying to convert 'std::any' into 'std::string'.");
    variable = encoded;
    return Result::SUCCESS;
}

Result Parser::StringToTypedValue(const std::string& encoded, I8& variable) {
    JST_TRACE("Deserializing: Trying to convert 'std::any' into 'I8'.");
    return StringToInteger(encoded, variable, "I8");
}

Result Parser::StringToTypedValue(const std::string& encoded, I16& variable) {
    JST_TRACE("Deserializing: Trying to convert 'std::any' into 'I16'.");
    return StringToInteger(encoded, variable, "I16");
}

Result Parser::StringToTypedValue(const std::string& encoded, I32& variable) {
    JST_TRACE("Deserializing: Trying to convert 'std::any' into 'I32'.");
    return StringToInteger(encoded, variable, "I32");
}

Result Parser::StringToTypedValue(const std::string& encoded, U8& variable) {
    JST_TRACE("Deserializing: Trying to convert 'std::any' into 'U8'.");
    return StringToInteger(encoded, variable, "U8");
}

Result Parser::StringToTypedValue(const std::string& encoded, U16& variable) {
    JST_TRACE("Deserializing: Trying to convert 'std::any' into 'U16'.");
    return StringToInteger(encoded, variable, "U16");
}

Result Parser::StringToTypedValue(const std::string& encoded, U32& variable) {
    JST_TRACE("Deserializing: Trying to convert 'std::any' into 'U32'.");
    return StringToInteger(encoded, variable, "U32");
}

Result Parser::StringToTypedValue(const std::string& encoded, I64& variable) {
    JST_TRACE("Deserializing: Trying to convert 'std::any' into 'I64'.");
    return StringToInteger(encoded, variable, "I64");
}

Result Parser::StringToTypedValue(const std::string& encoded, U64& variable) {
    JST_TRACE("Deserializing: Trying to convert 'std::any' into 'U64'.");
    return StringToInteger(encoded, variable, "U64");
}

Result Parser::StringToTypedValue(const std::string& encoded, F32& variable) {
    JST_TRACE("Deserializing: Trying to convert 'std::any' into 'F32'.");
    return StringToFloating(encoded, variable, "F32");
}

Result Parser::StringToTypedValue(const std::string& encoded, F64& variable) {
    JST_TRACE("Deserializing: Trying to convert 'std::any' into 'F64'.");
    return StringToFloating(encoded, variable, "F64");
}

Result Parser::StringToTypedValue(const std::string& encoded, CF32& variable) {
    JST_TRACE("Deserializing: Trying to convert 'std::any' into 'CF32'.");
    return StringToComplex(encoded, variable);
}

Result Parser::StringToTypedValue(const std::string& encoded, CF64& variable) {
    JST_TRACE("Deserializing: Trying to convert 'std::any' into 'CF64'.");
    return StringToComplex(encoded, variable);
}

Result Parser::StringToTypedValue(const std::string& encoded, bool& variable) {
    JST_TRACE("Deserializing: Trying to convert 'std::any' into 'bool'.");
    std::string normalized = encoded;
    std::transform(normalized.begin(), normalized.end(), normalized.begin(), [](const unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
    if (normalized == "true" || normalized == "1") {
        variable = true;
        return Result::SUCCESS;
    }
    if (normalized == "false" || normalized == "0") {
        variable = false;
        return Result::SUCCESS;
    }

    JST_ERROR("[PARSER] Value '{}' is not a valid 'bool'.", encoded);
    return Result::ERROR;
}

Result Parser::StringToTypedValue(const std::string& encoded, DeviceType& variable) {
    JST_TRACE("Deserializing: Trying to convert 'std::any' into 'DeviceType'.");
    variable = StringToDevice(encoded);
    return Result::SUCCESS;
}

Result Parser::StringToTypedValue(const std::string& encoded, RuntimeType& variable) {
    JST_TRACE("Deserializing: Trying to convert 'std::any' into 'RuntimeType'.");
    variable = StringToRuntime(encoded);
    return Result::SUCCESS;
}

Result Parser::StringToTypedValue(const std::string& encoded, SchedulerType& variable) {
    JST_TRACE("Deserializing: Trying to convert 'std::any' into 'SchedulerType'.");
    variable = StringToScheduler(encoded);
    return Result::SUCCESS;
}

Result Parser::StringToTypedValue(const std::string& encoded, std::vector<U64>& variable) {
    JST_TRACE("Deserializing: Trying to convert 'std::any' into 'std::vector<U64>'.");
    return StringToVector(encoded, variable);
}

Result Parser::StringToTypedValue(const std::string& encoded, std::vector<F64>& variable) {
    JST_TRACE("Deserializing: Trying to convert 'std::any' into 'std::vector<F64>'.");
    return StringToVector(encoded, variable);
}

Result Parser::StringToTypedValue(const std::string& encoded, std::vector<F32>& variable) {
    JST_TRACE("Deserializing: Trying to convert 'std::any' into 'std::vector<F32>'.");
    return StringToVector(encoded, variable);
}

Result Parser::StringToTypedValue(const std::string& encoded, std::vector<CF32>& variable) {
    JST_TRACE("Deserializing: Trying to convert 'std::any' into 'std::vector<CF32>'.");
    return StringToVector(encoded, variable);
}

Result Parser::StringToTypedValue(const std::string& encoded, std::vector<CF64>& variable) {
    JST_TRACE("Deserializing: Trying to convert 'std::any' into 'std::vector<CF64>'.");
    return StringToVector(encoded, variable);
}

Result Parser::StringToTypedValue(const std::string& encoded, Range<F32>& variable) {
    JST_TRACE("Deserializing: Trying to convert 'std::any' into 'Range<F32>'.");
    const auto values = ParseListValues(encoded);
    JST_ASSERT(values.size() == 2, "Unexpected number of values.");
    Range<F32> candidate{};
    JST_CHECK(Parser::StringToTyped(values[0], candidate.min));
    JST_CHECK(Parser::StringToTyped(values[1], candidate.max));
    variable = candidate;
    return Result::SUCCESS;
}

Result Parser::StringToTypedValue(const std::string& encoded, Extent2D<U64>& variable) {
    JST_TRACE("Deserializing: Trying to convert 'std::any' into 'Extent2D<U64>'.");
    const auto values = ParseListValues(encoded);
    JST_ASSERT(values.size() == 2, "Unexpected number of values.");
    Extent2D<U64> candidate{};
    JST_CHECK(Parser::StringToTyped(values[0], candidate.x));
    JST_CHECK(Parser::StringToTyped(values[1], candidate.y));
    variable = candidate;
    return Result::SUCCESS;
}

Result Parser::StringToTypedValue(const std::string& encoded, Extent2D<F32>& variable) {
    JST_TRACE("Deserializing: Trying to convert 'std::any' into 'Extent2D<F32>'.");
    const auto values = ParseListValues(encoded);
    JST_ASSERT(values.size() == 2, "Unexpected number of values.");
    Extent2D<F32> candidate{};
    JST_CHECK(Parser::StringToTyped(values[0], candidate.x));
    JST_CHECK(Parser::StringToTyped(values[1], candidate.y));
    variable = candidate;
    return Result::SUCCESS;
}

}  // namespace Jetstream
