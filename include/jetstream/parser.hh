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
struct is_string_keyed_unordered_map : std::false_type {};

template<typename V, typename Hash, typename KeyEqual, typename Alloc>
struct is_string_keyed_unordered_map<std::unordered_map<std::string, V, Hash, KeyEqual, Alloc>>
    : std::bool_constant<!std::is_same_v<V, std::any>> {};

template<typename T>
concept StringKeyedUnorderedMap = is_string_keyed_unordered_map<std::remove_cvref_t<T>>::value;

template<typename>
struct is_vector : std::false_type {};

template<typename V, typename Alloc>
struct is_vector<std::vector<V, Alloc>> : std::true_type {};

template<typename T>
concept Vector = is_vector<std::remove_cvref_t<T>>::value;

template<typename>
inline constexpr bool always_false_v = false;

}  // namespace detail

class JETSTREAM_API Parser {
 public:
    typedef std::unordered_map<std::string, std::any> Map;
    typedef std::vector<std::any> Sequence;

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
            std::any encoded;
            JST_CHECK(Encode(variable, encoded));
            map[name] = std::move(encoded);
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

        return Decode(encoded, name, variable);
    }

    template<typename T>
    static std::size_t Hash(const T& variable) {
        using ValueType = std::remove_cvref_t<T>;

        if constexpr (detail::HasMemberHash<ValueType>) {
            return static_cast<std::size_t>(variable.hash());
        } else if constexpr (std::is_same_v<ValueType, Map>) {
            std::vector<std::string> keys;
            keys.reserve(variable.size());

            for (const auto& [key, _] : variable) {
                keys.push_back(key);
            }

            std::sort(keys.begin(), keys.end());

            std::size_t seed = variable.size();
            for (const auto& key : keys) {
                seed ^= std::hash<std::string>{}(key) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
                seed ^= Hash(variable.at(key)) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
            }

            return seed;
        } else if constexpr (std::is_same_v<ValueType, Sequence>) {
            std::size_t seed = variable.size();
            for (const auto& entry : variable) {
                seed ^= Hash(entry) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
            }

            return seed;
        } else if constexpr (std::is_same_v<ValueType, std::any>) {
            if (!variable.has_value()) {
                return 0;
            }

            if (variable.type() == typeid(Map)) {
                return Hash(std::any_cast<const Map&>(variable));
            }

            if (variable.type() == typeid(Sequence)) {
                return Hash(std::any_cast<const Sequence&>(variable));
            }

            std::string encoded;
            if (TypedToString(variable, encoded) != Result::SUCCESS) {
                JST_ERROR("[PARSER] Failed to hash 'std::any' value.");
                return 0;
            }

            return std::hash<std::string>{}(encoded);
        } else if constexpr (detail::StringKeyedUnorderedMap<ValueType>) {
            std::vector<std::string> keys;
            keys.reserve(variable.size());

            for (const auto& [key, _] : variable) {
                keys.push_back(key);
            }

            std::sort(keys.begin(), keys.end());

            std::size_t seed = variable.size();
            for (const auto& key : keys) {
                seed ^= std::hash<std::string>{}(key) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
                seed ^= Hash(variable.at(key)) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
            }

            return seed;
        } else if constexpr (detail::Vector<ValueType>) {
            std::size_t seed = variable.size();
            for (const auto& entry : variable) {
                seed ^= Hash(entry) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
            }

            return seed;
        } else if constexpr (detail::HasStdHash<ValueType>) {
            return std::hash<ValueType>{}(variable);
        } else {
            static_assert(detail::always_false_v<ValueType>, "[PARSER] Missing hash support for field type.");
        }
    }

    static Result YamlEncode(const Map& data, std::string& yaml);
    static Result YamlDecode(const std::string& yaml, Map& data);

    static std::vector<std::string> SplitString(const std::string& str, const std::string& delimiter);

 private:
    template<typename T>
    static Result Encode(const T& variable, std::any& encoded) {
        using ValueType = std::remove_cvref_t<T>;

        if constexpr (detail::StringKeyedUnorderedMap<ValueType>) {
            Map nested;
            for (const auto& [entryName, entryValue] : variable) {
                JST_CHECK(Serialize(nested, entryName, entryValue));
            }
            encoded = std::move(nested);
        } else if constexpr (detail::Vector<ValueType>) {
            using EntryType = typename ValueType::value_type;

            if constexpr (detail::HasParserSerialize<EntryType> ||
                          detail::StringKeyedUnorderedMap<EntryType> ||
                          detail::Vector<EntryType>) {
                Sequence sequence;
                sequence.reserve(variable.size());

                for (const auto& entry : variable) {
                    std::any nested;
                    JST_CHECK(Encode(entry, nested));
                    sequence.push_back(std::move(nested));
                }

                encoded = std::move(sequence);
            } else {
                encoded = variable;
            }
        } else if constexpr (detail::HasParserSerialize<ValueType>) {
            Map nested;
            JST_CHECK(variable.serialize(nested));
            encoded = std::move(nested);
        } else {
            encoded = variable;
        }

        return Result::SUCCESS;
    }

    template<typename T>
    static Result Decode(const std::any& encoded, const std::string& name, T& variable) {
        using ValueType = std::remove_cvref_t<T>;

        if (encoded.type() == typeid(ValueType)) {
            JST_TRACE("Deserializing '{}': Trying to convert 'std::any' into 'T'.", name);

            variable = std::any_cast<const ValueType&>(encoded);
            return Result::SUCCESS;
        }

        if constexpr (std::is_same_v<ValueType, Map>) {
            if (encoded.type() == typeid(Map)) {
                JST_TRACE("Deserializing '{}': Trying to convert 'std::any' into 'Parser::Map'.", name);

                variable = std::any_cast<const Map&>(encoded);
                return Result::SUCCESS;
            }

            JST_ERROR("[PARSER] Variable '{}' is not of type 'Parser::Map'.", name);
            return Result::ERROR;
        } else if constexpr (std::is_same_v<ValueType, Sequence>) {
            if (encoded.type() == typeid(Sequence)) {
                JST_TRACE("Deserializing '{}': Trying to convert 'std::any' into 'Parser::Sequence'.", name);

                variable = std::any_cast<const Sequence&>(encoded);
                return Result::SUCCESS;
            }

            JST_ERROR("[PARSER] Variable '{}' is not of type 'Parser::Sequence'.", name);
            return Result::ERROR;
        } else if constexpr (detail::StringKeyedUnorderedMap<ValueType>) {
            if (encoded.type() == typeid(Map)) {
                JST_TRACE("Deserializing '{}': Trying to convert nested 'Parser::Map' into unordered map.", name);

                const auto& nested = std::any_cast<const Map&>(encoded);
                ValueType decoded;

                for (const auto& [entryName, _] : nested) {
                    typename ValueType::mapped_type entryValue{};
                    JST_CHECK(Deserialize(nested, entryName, entryValue));
                    decoded.emplace(entryName, std::move(entryValue));
                }

                variable = std::move(decoded);
                return Result::SUCCESS;
            }

            if (encoded.type() == typeid(std::string)) {
                JST_ERROR("[PARSER] Variable '{}' cannot be deserialized from a string.", name);
                return Result::ERROR;
            }

            JST_ERROR("[PARSER] Variable '{}' is not of type 'Parser::Map'.", name);
            return Result::ERROR;
        } else if constexpr (detail::Vector<ValueType>) {
            using EntryType = typename ValueType::value_type;

            if (encoded.type() == typeid(Sequence)) {
                JST_TRACE("Deserializing '{}': Trying to convert nested 'Parser::Sequence' into vector.", name);

                const auto& sequence = std::any_cast<const Sequence&>(encoded);
                ValueType decoded;
                decoded.reserve(sequence.size());

                for (const auto& entry : sequence) {
                    EntryType decodedEntry{};
                    JST_CHECK(Decode(entry, name, decodedEntry));
                    decoded.push_back(std::move(decodedEntry));
                }

                variable = std::move(decoded);
                return Result::SUCCESS;
            }

            if (encoded.type() == typeid(std::string)) {
                if constexpr (std::is_same_v<ValueType, std::vector<U64>> ||
                              std::is_same_v<ValueType, std::vector<F32>> ||
                              std::is_same_v<ValueType, std::vector<F64>>) {
                    JST_CHECK(StringToTyped<ValueType>(std::any_cast<const std::string&>(encoded), variable));
                    return Result::SUCCESS;
                }

                JST_ERROR("[PARSER] Variable '{}' cannot be deserialized from a string.", name);
                return Result::ERROR;
            }

            JST_ERROR("[PARSER] Variable '{}' is not of type 'Parser::Sequence'.", name);
            return Result::ERROR;
        } else if constexpr (detail::HasParserDeserialize<ValueType>) {
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

            JST_CHECK(StringToTyped<ValueType>(std::any_cast<const std::string&>(encoded), variable));

            return Result::SUCCESS;
        }
    }
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
struct hash<Jetstream::Parser::Map> {
    std::size_t operator()(const Jetstream::Parser::Map& value) const noexcept {
        return Jetstream::Parser::Hash(value);
    }
};

template<>
struct hash<Jetstream::Parser::Sequence> {
    std::size_t operator()(const Jetstream::Parser::Sequence& value) const noexcept {
        return Jetstream::Parser::Hash(value);
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
