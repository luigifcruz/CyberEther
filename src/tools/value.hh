#ifndef JETSTREAM_TOOLS_VALUE_HH
#define JETSTREAM_TOOLS_VALUE_HH

#include "jetstream/parser.hh"
#include "tools/text.hh"
#include "tools/vector_types.hh"

#include <type_traits>

namespace Jetstream::Value {

inline bool KeyMatches(const std::string& key, const std::string& filter) {
    return filter.empty() || Text::ToLower(key).find(filter) != std::string::npos;
}

inline std::string AnyToString(const std::any& value);

template<typename Format>
inline std::string MapToString(const Parser::Map& map, const Format& format) {
    std::vector<std::string> values;
    for (const auto& entry : map) {
        values.push_back(jst::fmt::format("{}: {}", entry.key, format(entry.value)));
    }
    return jst::fmt::format("{{{}}}", jst::fmt::join(values, ", "));
}

inline std::string MapToString(const Parser::Map& map) {
    return MapToString(map, AnyToString);
}

template<typename T>
inline std::string ComplexToString(const T& complex) {
    return jst::fmt::format("{}{}{}i", complex.real(), complex.imag() < 0 ? "" : "+", complex.imag());
}

template<typename T>
inline std::string ElementToString(const T& value) {
    if constexpr (VectorTypes::IsComplex<T>) {
        return ComplexToString(value);
    } else if constexpr (std::is_same_v<T, Parser::Map>) {
        return MapToString(value);
    } else {
        return jst::fmt::format("{}", value);
    }
}

inline std::string AnyToString(const std::any& value) {
    if (!value.has_value()) {
        return "null";
    }
    if (value.type() == typeid(Parser::Map)) {
        return MapToString(std::any_cast<const Parser::Map&>(value));
    }
    if (value.type() == typeid(Parser::Sequence)) {
        std::vector<std::string> values;
        for (const auto& entry : std::any_cast<const Parser::Sequence&>(value)) {
            values.push_back(AnyToString(entry));
        }
        return jst::fmt::format("[{}]", jst::fmt::join(values, ", "));
    }
    if (value.type() == typeid(CF32)) {
        return ComplexToString(std::any_cast<CF32>(value));
    }
    if (value.type() == typeid(CF64)) {
        return ComplexToString(std::any_cast<CF64>(value));
    }

    std::string encoded;
    const bool vector = VectorTypes::Visit(value, [&](const auto& entries) {
        using T = typename std::decay_t<decltype(entries)>::value_type;
        std::vector<std::string> values;
        values.reserve(entries.size());
        for (const auto& entry : entries) {
            values.push_back(ElementToString(T(entry)));
        }
        encoded = jst::fmt::format("[{}]", jst::fmt::join(values, ", "));
    });
    if (vector || Parser::TryTypedToString(value, encoded)) {
        return encoded;
    }
    return "?";
}

}  // namespace Jetstream::Value

#endif  // JETSTREAM_TOOLS_VALUE_HH
