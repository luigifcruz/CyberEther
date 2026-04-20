#ifndef JETSTREAM_PARSER_DETAIL_HH
#define JETSTREAM_PARSER_DETAIL_HH

#include <concepts>
#include <functional>
#include <type_traits>
#include <unordered_map>
#include <vector>

#include "jetstream/parser_map.hh"
#include "jetstream/types.hh"

namespace Jetstream::detail {

template<typename T>
concept HasParserSerialize = requires(const std::remove_cvref_t<T>& value, ParserMap& data) {
    { value.serialize(data) } -> std::same_as<Result>;
};

template<typename T>
concept HasParserDeserialize = requires(std::remove_cvref_t<T>& value, const ParserMap& data) {
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

}  // namespace Jetstream::detail

#endif  // JETSTREAM_PARSER_DETAIL_HH
