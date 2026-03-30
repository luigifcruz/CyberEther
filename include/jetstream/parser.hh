#ifndef JETSTREAM_PARSER_HH
#define JETSTREAM_PARSER_HH

#include <string>
#include <vector>
#include <memory>
#include <functional>
#include <algorithm>
#include <numeric>
#include <iterator>
#include <sstream>
#include <iostream>
#include <optional>
#include <unordered_map>
#include <unordered_set>

#include "jetstream/types.hh"
#include "jetstream/macros.hh"
#include "jetstream/logger.hh"
#include "jetstream/memory/tensor.hh"

namespace Jetstream {

struct BlockEndpoint {
    std::string block;
    std::string port;
};

struct ModuleEndpoint {
    std::string module;
    std::string port;
};

struct TensorLink {
    std::optional<ModuleEndpoint> producer;
    std::optional<BlockEndpoint> external;
    Tensor tensor;

    void requested(const std::string& block, const std::string& port);
    void produced(const std::string& module, const std::string& port, const Tensor& tensor);
    void exposedAs(const std::string& block, const std::string& port);

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
    static Result Serialize(Map& map, const std::string& name, T& variable) {
        if (map.contains(name) != 0) {
            JST_TRACE("Variable name '{}' already inside map. Overwriting.", name);
            map.erase(name);
        }

        try {
            map[name] = std::any(variable);
        } catch (const std::bad_any_cast& e) {
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

        if (encoded.type() != typeid(std::string)) {
            JST_ERROR("[PARSER] Variable '{}' is not of type 'std::string'.", name);
            return Result::ERROR;
        }

        JST_CHECK(StringToTyped<T>(std::any_cast<std::string>(encoded), variable));

        return Result::SUCCESS;
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
    h ^= std::hash<std::decay_t<decltype(field)>>{}(field) + 0x9e3779b9 + (h << 6) + (h >> 2);
#endif  // JST_HASH_FIELD

#endif
