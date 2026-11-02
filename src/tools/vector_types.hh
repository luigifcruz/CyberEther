#ifndef JETSTREAM_TOOLS_VECTOR_TYPES_HH
#define JETSTREAM_TOOLS_VECTOR_TYPES_HH

#include "jetstream/parser_map.hh"
#include "jetstream/types.hh"

#include <any>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

namespace Jetstream::VectorTypes {

template<typename T>
inline constexpr bool IsComplex = std::is_same_v<T, CF32> || std::is_same_v<T, CF64>;

template<typename T>
constexpr std::string_view Name() {
    if constexpr (std::is_same_v<T, std::string>) return "string";
    else if constexpr (std::is_same_v<T, bool>) return "bool";
    else if constexpr (std::is_same_v<T, I8>) return "I8";
    else if constexpr (std::is_same_v<T, U8>) return "U8";
    else if constexpr (std::is_same_v<T, I16>) return "I16";
    else if constexpr (std::is_same_v<T, U16>) return "U16";
    else if constexpr (std::is_same_v<T, I32>) return "I32";
    else if constexpr (std::is_same_v<T, U32>) return "U32";
    else if constexpr (std::is_same_v<T, I64>) return "I64";
    else if constexpr (std::is_same_v<T, U64>) return "U64";
    else if constexpr (std::is_same_v<T, F32>) return "F32";
    else if constexpr (std::is_same_v<T, F64>) return "F64";
    else if constexpr (std::is_same_v<T, CF32>) return "CF32";
    else if constexpr (std::is_same_v<T, CF64>) return "CF64";
    else return "map";
}

template<typename... Types, typename Visitor>
bool VisitAs(const std::any& value, Visitor&& visitor) {
    return ((value.type() == typeid(std::vector<Types>) &&
             (visitor(std::any_cast<const std::vector<Types>&>(value)), true)) || ...);
}

template<typename Visitor>
bool Visit(const std::any& value, Visitor&& visitor) {
    return VisitAs<std::string, bool, I8, U8, I16, U16, I32, U32, I64, U64, F32, F64, CF32, CF64,
                   ParserMap>(value, visitor);
}

}  // namespace Jetstream::VectorTypes

#endif  // JETSTREAM_TOOLS_VECTOR_TYPES_HH
