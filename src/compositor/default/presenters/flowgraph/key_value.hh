#ifndef JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_FLOWGRAPH_KEY_VALUE_HH
#define JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_FLOWGRAPH_KEY_VALUE_HH

#include "jetstream/parser.hh"

#include <algorithm>
#include <any>
#include <cctype>
#include <string>
#include <vector>

namespace Jetstream::FlowgraphKeyValueDetail {

inline std::string AnyToString(const std::any& value);

inline std::string NormalizeFilter(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](const unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
    return value;
}

inline bool KeyMatches(const std::string& key, const std::string& filter) {
    if (filter.empty()) {
        return true;
    }
    return NormalizeFilter(key).find(filter) != std::string::npos;
}

inline std::string EntryCount(const U64 shown, const U64 total) {
    return jst::fmt::format("{} of {} entr{}", shown, total, total == 1 ? "y" : "ies");
}

template<typename T>
inline bool NumericToString(const std::any& value, std::string& encoded) {
    if (value.type() != typeid(T)) {
        return false;
    }
    encoded = jst::fmt::format("{}", std::any_cast<T>(value));
    return true;
}

template<typename T>
inline bool UnsignedToString(const std::any& value, std::string& encoded) {
    if (value.type() != typeid(T)) {
        return false;
    }
    encoded = jst::fmt::format("{}", static_cast<U64>(std::any_cast<T>(value)));
    return true;
}

template<typename T>
inline bool SignedToString(const std::any& value, std::string& encoded) {
    if (value.type() != typeid(T)) {
        return false;
    }
    encoded = jst::fmt::format("{}", static_cast<I64>(std::any_cast<T>(value)));
    return true;
}

template<typename T>
inline std::string VectorToString(const std::vector<T>& values) {
    return jst::fmt::format("[{}]", jst::fmt::join(values, ", "));
}

inline std::string SequenceToString(const Parser::Sequence& sequence) {
    std::vector<std::string> values;
    values.reserve(sequence.size());
    for (const auto& entry : sequence) {
        values.push_back(AnyToString(entry));
    }
    return jst::fmt::format("[{}]", jst::fmt::join(values, ", "));
}

inline std::string MapToString(const Parser::Map& map) {
    std::vector<std::string> values;
    values.reserve(map.size());
    for (const auto& entry : map) {
        values.push_back(jst::fmt::format("{}: {}", entry.key, AnyToString(entry.value)));
    }
    return jst::fmt::format("{{{}}}", jst::fmt::join(values, ", "));
}

inline std::string AnyToString(const std::any& value) {
    if (!value.has_value()) {
        return "null";
    }

    if (value.type() == typeid(Parser::Map)) {
        return MapToString(std::any_cast<const Parser::Map&>(value));
    }
    if (value.type() == typeid(Parser::Sequence)) {
        return SequenceToString(std::any_cast<const Parser::Sequence&>(value));
    }
    if (value.type() == typeid(std::string)) {
        return std::any_cast<const std::string&>(value);
    }
    if (value.type() == typeid(bool)) {
        return std::any_cast<bool>(value) ? "true" : "false";
    }

    std::string encoded;
    if (SignedToString<I8>(value, encoded) ||
        SignedToString<I16>(value, encoded) ||
        SignedToString<I32>(value, encoded) ||
        SignedToString<I64>(value, encoded) ||
        UnsignedToString<U8>(value, encoded) ||
        UnsignedToString<U16>(value, encoded) ||
        UnsignedToString<U32>(value, encoded) ||
        UnsignedToString<U64>(value, encoded) ||
        NumericToString<F32>(value, encoded) ||
        NumericToString<F64>(value, encoded)) {
        return encoded;
    }

    if (value.type() == typeid(CF32)) {
        const auto& complex = std::any_cast<const CF32&>(value);
        return jst::fmt::format("{}{}{}i", complex.real(), complex.imag() < 0 ? "" : "+", complex.imag());
    }
    if (value.type() == typeid(CF64)) {
        const auto& complex = std::any_cast<const CF64&>(value);
        return jst::fmt::format("{}{}{}i", complex.real(), complex.imag() < 0 ? "" : "+", complex.imag());
    }
    if (value.type() == typeid(std::vector<std::string>)) {
        return VectorToString(std::any_cast<const std::vector<std::string>&>(value));
    }
    if (value.type() == typeid(std::vector<U64>)) {
        return VectorToString(std::any_cast<const std::vector<U64>&>(value));
    }
    if (value.type() == typeid(std::vector<I64>)) {
        return VectorToString(std::any_cast<const std::vector<I64>&>(value));
    }
    if (value.type() == typeid(std::vector<F32>)) {
        return VectorToString(std::any_cast<const std::vector<F32>&>(value));
    }
    if (value.type() == typeid(std::vector<F64>)) {
        return VectorToString(std::any_cast<const std::vector<F64>&>(value));
    }

    if (value.type() == typeid(DeviceType) ||
        value.type() == typeid(RuntimeType) ||
        value.type() == typeid(SchedulerType) ||
        value.type() == typeid(Range<F32>) ||
        value.type() == typeid(Extent2D<U64>) ||
        value.type() == typeid(Extent2D<F32>)) {
        if (Parser::TypedToString(value, encoded) == Result::SUCCESS) {
            return encoded;
        }
    }

    return "?";
}

}  // namespace Jetstream::FlowgraphKeyValueDetail

#endif  // JETSTREAM_COMPOSITOR_IMPL_DEFAULT_PRESENTERS_FLOWGRAPH_KEY_VALUE_HH
