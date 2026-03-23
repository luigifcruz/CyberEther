#ifndef JETSTREAM_PARSER_HH
#define JETSTREAM_PARSER_HH

#include <algorithm>
#include <any>
#include <concepts>
#include <functional>
#include <iostream>
#include <iterator>
#include <memory>
#include <numeric>
#include <sstream>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "jetstream/types.hh"
#include "jetstream/macros.hh"
#include "jetstream/logger.hh"
#include "jetstream/memory/tensor.hh"

namespace Jetstream {

namespace detail {

using ParserMapType = std::unordered_map<std::string, std::any>;

template<typename T>
concept HasParserSerialize = requires(const std::remove_cvref_t<T>& value, ParserMapType& data) {
    { value.serialize(data) } -> std::same_as<Result>;
};

template<typename T>
concept HasParserDeserialize = requires(std::remove_cvref_t<T>& value, const ParserMapType& data) {
    { value.deserialize(data) } -> std::same_as<Result>;
};

template<typename T>
concept HasMemberHash = requires(const std::remove_cvref_t<T>& value) {
    { value.hash() } -> std::convertible_to<std::size_t>;
};

template<typename T>
concept HasStdHash = requires(const std::remove_cvref_t<T>& value) {
    { std::hash<std::remove_cvref_t<T>>{}(value) } -> std::convertible_to<std::size_t>;
};

template<typename>
inline constexpr bool always_false_v = false;

}  // namespace detail

struct TensorLink {
    std::string block;
    std::string port;
    Tensor tensor;

    bool resolved() const;
};

typedef std::unordered_map<std::string, TensorLink> TensorMap;

class JETSTREAM_API Parser {
 public:
    typedef std::unordered_map<std::string, std::any> Map;

    template<typename T>
    static Result StringToTyped(const std::string& encoded, T& variable);
    static Result TypedToString(const std::any& variable, std::string& encoded);

    template<typename T>
    static Result Serialize(Map& map, const std::string& name, const T& variable) {
        if (map.contains(name) != 0) {
            JST_TRACE("Variable name '{}' already inside map. Overwriting.", name);
            map.erase(name);
        }

        try {
            if constexpr (detail::HasParserSerialize<T>) {
                Map nested;
                JST_CHECK(variable.serialize(nested));
                map[name] = std::move(nested);
            } else {
                map[name] = std::any(variable);
            }
        } catch (const std::exception& e) {
            JST_ERROR("[PARSER] Failed to serialize variable '{}': {}", name, e.what());
            return Result::ERROR;
        }

        return Result::SUCCESS;
    }

    template<typename T>
    static Result Deserialize(const Map& map, const std::string& name, T& variable) {
        if (map.contains(name) == 0) {
            JST_TRACE("[PARSER] Variable name '{}' not found inside map.", name);
            return Result::SUCCESS;
        }

        const auto& encoded = map.at(name);
        if (!encoded.has_value()) {
            JST_ERROR("[PARSER] Variable '{}' not initialized.", name);
            return Result::ERROR;
        }

        if (encoded.type() == typeid(T)) {
            JST_TRACE("Deserializing '{}': Trying to convert 'std::any' into 'T'.", name);

            variable = std::move(std::any_cast<T>(encoded));
            return Result::SUCCESS;
        }

        if constexpr (detail::HasParserDeserialize<T>) {
            if (encoded.type() == typeid(Map)) {
                JST_TRACE("Deserializing '{}': Trying to convert nested 'Parser::Map' into 'T'.", name);
                return variable.deserialize(std::any_cast<const Map&>(encoded));
            }

            if (encoded.type() == typeid(std::string)) {
                JST_ERROR("[PARSER] Variable '{}' cannot be deserialized from a string.", name);
                return Result::ERROR;
            }

            JST_ERROR("[PARSER] Variable '{}' is not of type 'Parser::Map'.", name);
            return Result::ERROR;
        } else {
            if (encoded.type() != typeid(std::string)) {
                JST_ERROR("[PARSER] Variable '{}' is not of type 'std::string'.", name);
                return Result::ERROR;
            }

            JST_CHECK(StringToTyped<T>(std::any_cast<std::string>(encoded), variable));

            return Result::SUCCESS;
        }
    }

    template<typename T>
    static std::size_t Hash(const T& variable) {
        using ValueType = std::remove_cvref_t<T>;

        if constexpr (detail::HasMemberHash<ValueType>) {
            return static_cast<std::size_t>(variable.hash());
        } else if constexpr (detail::HasStdHash<ValueType>) {
            return std::hash<ValueType>{}(variable);
        } else {
            static_assert(detail::always_false_v<ValueType>, "[PARSER] Missing hash support for field type.");
        }
    }

    static std::vector<std::string> SplitString(const std::string& str, const std::string& delimiter);
};

}  // namespace Jetstream

#ifndef JST_SERDES_SERIALIZE
#define JST_SERDES_SERIALIZE(var) \
    JST_CHECK(Parser::Serialize(data, #var, var));
#endif  // JST_SERDES_SERIALIZE

#ifndef JST_SERDES_DESERIALIZE
#define JST_SERDES_DESERIALIZE(var) \
    JST_CHECK(Parser::Deserialize(data, #var, var));
#endif  // JST_SERDES_DESERIALIZE

#ifndef JST_SERDES
#define JST_SERDES(...) \
    Result serialize(Parser::Map& data) const { \
        (void)data; \
        FOR_EACH(JST_SERDES_SERIALIZE, __VA_ARGS__) \
        return Result::SUCCESS; \
    } \
    Result deserialize(const Parser::Map& data) { \
        (void)data; \
        FOR_EACH(JST_SERDES_DESERIALIZE, __VA_ARGS__) \
        return Result::SUCCESS; \
    } \
    std::size_t hash() const { \
        std::size_t h = 0; \
        FOR_EACH(JST_HASH_FIELD, __VA_ARGS__) \
        return h; \
    }
#endif  // JST_SERDES

namespace std {

template<typename T>
struct hash<std::vector<T>> {
    std::size_t operator()(const std::vector<T>& v) const noexcept {
        std::size_t seed = v.size();
        for (const auto& elem : v) {
            seed ^= std::hash<T>{}(elem) + 0x9e3779b9 +
                    (seed << 6) + (seed >> 2);
        }
        return seed;
    }
};

template<>
struct hash<Jetstream::Tensor> {
    std::size_t operator()(const Jetstream::Tensor& t) const noexcept {
        return std::hash<Jetstream::U64>{}(t.id());
    }
};

}  // namespace std

#ifndef JST_HASH_FIELD
#define JST_HASH_FIELD(field) \
    h ^= Parser::Hash(field) + 0x9e3779b9 + (h << 6) + (h >> 2);
#endif  // JST_HASH_FIELD

#endif
