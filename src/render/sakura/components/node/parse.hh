#ifndef JETSTREAM_RENDER_SAKURA_NODE_PARSE_HH
#define JETSTREAM_RENDER_SAKURA_NODE_PARSE_HH

#include <jetstream/types.hh>

#include "tools/text.hh"

#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <optional>
#include <string>
#include <utility>

namespace Jetstream::Sakura {

inline bool DecimalCharacter(U32 character) {
    return (character >= '0' && character <= '9') || character == '.' || character == '+' || character == '-';
}

inline bool ScientificCharacter(U32 character) {
    return DecimalCharacter(character) || character == 'e' || character == 'E';
}

inline std::optional<F64> ParseFloat(const std::string& text) {
    const std::string trimmed = Jetstream::Text::TrimBlank(text);
    if (trimmed.empty()) {
        return std::nullopt;
    }
    char* end = nullptr;
    errno = 0;
    const F64 value = std::strtod(trimmed.c_str(), &end);
    if (end == trimmed.c_str() || errno == ERANGE || !std::isfinite(value)) {
        return std::nullopt;
    }
    return value;
}

inline std::optional<F64> ParseScaledFloat(const std::string& text, const F64 scale) {
    const auto parsed = ParseFloat(text);
    if (!parsed.has_value()) {
        return std::nullopt;
    }
    const F64 decimals = std::round(std::log10(scale));
    if (!std::isfinite(scale) || scale <= 0.0 || std::pow(10.0, decimals) != scale) {
        return *parsed * scale;
    }

    const std::string trimmed = Jetstream::Text::TrimBlank(text);
    char* end = nullptr;
    std::strtod(trimmed.c_str(), &end);
    std::string number = trimmed.substr(0, static_cast<std::size_t>(end - trimmed.c_str()));
    long exponent = 0;
    if (const auto mark = number.find_first_of("eE"); mark != std::string::npos) {
        errno = 0;
        exponent = std::strtol(number.c_str() + mark + 1, nullptr, 10);
        if (errno == ERANGE || !std::in_range<I32>(exponent)) {
            return *parsed * scale;
        }
        number.resize(mark);
    }
    exponent += static_cast<long>(decimals);

    errno = 0;
    const F64 value = std::strtod((number + "e" + std::to_string(exponent)).c_str(), nullptr);
    if (errno == ERANGE || !std::isfinite(value)) {
        return *parsed * scale;
    }
    return value;
}

}  // namespace Jetstream::Sakura

#endif  // JETSTREAM_RENDER_SAKURA_NODE_PARSE_HH
