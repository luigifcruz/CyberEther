#include "tools/json.hh"
#include "tools/vector_types.hh"

#include <array>
#include <cmath>
#include <optional>
#include <string>

namespace Jetstream::Value {
namespace {

using Json = nlohmann::json;
using Encoder = std::optional<Result> (*)(const std::any& value, Json& encoded, bool log);

Result Reject(const bool log, const std::string& message) {
    if (log) {
        JST_ERROR("[JSON] {}", message);
    }
    return Result::ERROR;
}

template<typename Function>
Result Guard(const bool log, Function&& convert) {
    try {
        return convert();
    } catch (const std::exception& error) {
        return Reject(log, std::string("JSON conversion failed: ") + error.what());
    } catch (...) {
        return Reject(log, "JSON conversion failed with an unknown error.");
    }
}

Result EncodeValue(const std::any& value, Json& encoded, bool log);

Result EncodeMap(const Parser::Map& map, Json& encoded, const bool log) {
    encoded = Json::object();
    for (const auto& entry : map) {
        JST_CHECK(EncodeValue(entry.value, encoded[entry.key], log));
    }
    return Result::SUCCESS;
}

Result EncodeSequence(const Parser::Sequence& sequence, Json& encoded, const bool log) {
    encoded = Json::array();
    for (const auto& entry : sequence) {
        JST_CHECK(EncodeValue(entry, encoded.emplace_back(), log));
    }
    return Result::SUCCESS;
}

template<typename T>
std::optional<Result> EncodeScalar(const std::any& value, Json& encoded, const bool log) {
    const auto* scalar = std::any_cast<T>(&value);
    if (!scalar) {
        return std::nullopt;
    }
    if constexpr (std::is_floating_point_v<T>) {
        if (!std::isfinite(*scalar)) {
            return Reject(log, "JSON numbers must be finite.");
        }
    }
    encoded = *scalar;
    return Result::SUCCESS;
}

std::optional<Result> EncodeVector(const std::any& value, Json& encoded, const bool log) {
    std::optional<Result> status;
    VectorTypes::Visit(value, [&](const auto& entries) {
        using T = typename std::decay_t<decltype(entries)>::value_type;
        if constexpr (VectorTypes::IsComplex<T>) {
            std::string text;
            if (!Parser::TryTypedToString(value, text)) {
                status = Reject(log, "Unsupported complex vector encoding.");
                return;
            }
            encoded = text;
            status = Result::SUCCESS;
        } else {
            encoded = Json::array();
            for (const auto& entry : entries) {
                status = EncodeValue(T(entry), encoded.emplace_back(), log);
                if (*status != Result::SUCCESS) {
                    return;
                }
            }
            status = Result::SUCCESS;
        }
    });
    return status;
}

constexpr auto kEncoders = std::to_array<Encoder>({
    EncodeScalar<std::string>, EncodeScalar<bool>,
    EncodeScalar<I8>, EncodeScalar<U8>, EncodeScalar<I16>, EncodeScalar<U16>,
    EncodeScalar<I32>, EncodeScalar<U32>, EncodeScalar<I64>, EncodeScalar<U64>,
    EncodeScalar<F32>, EncodeScalar<F64>, EncodeVector,
});

Result EncodeValue(const std::any& value, Json& encoded, const bool log) {
    if (!value.has_value() || value.type() == typeid(std::nullptr_t)) {
        encoded = nullptr;
        return Result::SUCCESS;
    }
    if (const auto* map = std::any_cast<Parser::Map>(&value)) {
        return EncodeMap(*map, encoded, log);
    }
    if (const auto* sequence = std::any_cast<Parser::Sequence>(&value)) {
        return EncodeSequence(*sequence, encoded, log);
    }
    if (const auto* text = std::any_cast<const char*>(&value)) {
        encoded = std::string(*text ? *text : "");
        return Result::SUCCESS;
    }
    for (const auto encoder : kEncoders) {
        if (const auto status = encoder(value, encoded, log)) {
            return *status;
        }
    }

    if (std::string text; Parser::TryTypedToString(value, text)) {
        encoded = text;
        return Result::SUCCESS;
    }
    return Reject(log, "Unsupported JSON value type: " + std::string(value.type().name()));
}

Result EncodeAny(const std::any& value, Json& json, const bool log) {
    return Guard(log, [&] {
        Json encoded;
        JST_CHECK(EncodeValue(value, encoded, log));
        json = std::move(encoded);
        return Result::SUCCESS;
    });
}

Result DecodeValue(const Json& value, std::any& decoded);

Result DecodeMap(const Json& value, Parser::Map& map) {
    map.reserve(value.size());
    for (auto it = value.begin(); it != value.end(); ++it) {
        JST_CHECK(DecodeValue(it.value(), map[it.key()]));
    }
    return Result::SUCCESS;
}

Result DecodeSequence(const Json& value, Parser::Sequence& sequence) {
    sequence.reserve(value.size());
    for (const auto& entry : value) {
        JST_CHECK(DecodeValue(entry, sequence.emplace_back()));
    }
    return Result::SUCCESS;
}

Result DecodeScalar(const Json& value, std::any& decoded) {
    if (value.is_string()) {
        decoded = value.get<std::string>();
    } else if (value.is_boolean()) {
        decoded = value.get<bool>();
    } else if (value.is_number_unsigned()) {
        decoded = value.get<U64>();
    } else if (value.is_number_integer()) {
        decoded = value.get<I64>();
    } else if (value.is_null()) {
        decoded = std::nullptr_t{};
    } else {
        return Reject(true, "Unsupported JSON value type: " + std::string(value.type_name()));
    }
    return Result::SUCCESS;
}

Result DecodeValue(const Json& value, std::any& decoded) {
    if (value.is_object()) {
        Parser::Map map;
        JST_CHECK(DecodeMap(value, map));
        decoded = std::move(map);
        return Result::SUCCESS;
    }
    if (value.is_array()) {
        Parser::Sequence sequence;
        JST_CHECK(DecodeSequence(value, sequence));
        decoded = std::move(sequence);
        return Result::SUCCESS;
    }
    if (value.is_number_float()) {
        const auto number = value.get<F64>();
        if (!std::isfinite(number)) {
            return Reject(true, "JSON numbers must be finite.");
        }
        decoded = number;
        return Result::SUCCESS;
    }
    return DecodeScalar(value, decoded);
}

}  // namespace

Result ToJson(const Parser::Map& data, nlohmann::json& json) {
    return Guard(true, [&] {
        Json encoded;
        JST_CHECK(EncodeMap(data, encoded, true));
        json = std::move(encoded);
        return Result::SUCCESS;
    });
}

Result ToJson(const std::any& value, nlohmann::json& json) {
    return EncodeAny(value, json, true);
}

bool TryToJson(const std::any& value, nlohmann::json& json) {
    return EncodeAny(value, json, false) == Result::SUCCESS;
}

Result FromJson(const nlohmann::json& json, Parser::Map& data) {
    if (!json.is_object()) {
        return Reject(true, "JSON configuration must be an object.");
    }
    return Guard(true, [&] {
        Parser::Map decoded;
        JST_CHECK(DecodeMap(json, decoded));
        data = std::move(decoded);
        return Result::SUCCESS;
    });
}

}  // namespace Jetstream::Value
