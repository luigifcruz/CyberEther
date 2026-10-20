#ifndef JETSTREAM_RENDER_SAKURA_NODE_PARSE_HH
#define JETSTREAM_RENDER_SAKURA_NODE_PARSE_HH

#include <jetstream/types.hh>

#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <optional>
#include <string>

namespace Jetstream::Sakura {

inline std::string Trim(const std::string& value) {
    const auto begin = value.find_first_not_of(" \t\r\n");
    if (begin == std::string::npos) {
        return {};
    }
    const auto end = value.find_last_not_of(" \t\r\n");
    return value.substr(begin, end - begin + 1);
}

inline bool DecimalCharacter(U32 character) {
    return (character >= '0' && character <= '9') || character == '.' || character == '+' || character == '-';
}

inline bool ScientificCharacter(U32 character) {
    return DecimalCharacter(character) || character == 'e' || character == 'E';
}

inline std::optional<F64> ParseFloat(const std::string& text) {
    const std::string trimmed = Trim(text);
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

}  // namespace Jetstream::Sakura

#endif  // JETSTREAM_RENDER_SAKURA_NODE_PARSE_HH
