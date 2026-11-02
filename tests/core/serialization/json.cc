#include <catch2/catch_test_macros.hpp>

#include "fixture.hh"

#include "tools/json.hh"

using namespace Jetstream;
using namespace TestSerialization;

namespace {

Parser::Map DecodeJson(const nlohmann::json& json) {
    Parser::Map map;
    REQUIRE(Value::FromJson(json, map) == Result::SUCCESS);
    return map;
}

template<typename T>
nlohmann::json EncodeJson(const T& value) {
    nlohmann::json json;
    REQUIRE(Value::ToJson(value, json) == Result::SUCCESS);
    return json;
}

Result DecodeStatus(const nlohmann::json& json) {
    Parser::Map map;
    return Value::FromJson(json, map);
}

template<typename T>
Result EncodeStatus(const T& value) {
    nlohmann::json json;
    return Value::ToJson(value, json);
}

}  // namespace

TEST_CASE("Parser JSON preserves native scalars and 64-bit integers", "[core][serialization][json]") {
    const auto json = nlohmann::json::parse(R"({
        "signed": -9223372036854775808,
        "unsigned": 18446744073709551615,
        "fraction": 1.25,
        "enabled": false,
        "text": "true",
        "nested": {"count": 64}
    })");
    const auto map = DecodeJson(json);
    REQUIRE(std::any_cast<I64>(map.at("signed")) == std::numeric_limits<I64>::min());
    REQUIRE(std::any_cast<U64>(map.at("unsigned")) == std::numeric_limits<U64>::max());
    REQUIRE(std::any_cast<F64>(map.at("fraction")) == 1.25);
    REQUIRE_FALSE(std::any_cast<bool>(map.at("enabled")));
    REQUIRE(std::any_cast<std::string>(map.at("text")) == "true");
    const auto& nested = std::any_cast<const Parser::Map&>(map.at("nested"));
    REQUIRE(std::any_cast<U64>(nested.at("count")) == 64);

    const auto encoded = EncodeJson(map);
    REQUIRE(encoded == json);
    REQUIRE(encoded["signed"].get<I64>() == std::numeric_limits<I64>::min());
    REQUIRE(encoded["unsigned"].is_number_unsigned());
    REQUIRE(encoded["unsigned"].get<U64>() == std::numeric_limits<U64>::max());
    REQUIRE(nlohmann::json::parse(encoded.dump()) == json);
}

TEST_CASE("Parser JSON round-trips mixed arrays, nulls, and empty containers", "[core][serialization][json]") {
    const auto json = nlohmann::json::parse(R"({
        "": "empty key", "object": {}, "array": [],
        "items": [null, true, -2, 1.5, "quote: \"\nλ", {"id": 5}, [[], {}]]
    })");
    const auto map = DecodeJson(json);
    const auto& items = std::any_cast<const Parser::Sequence&>(map.at("items"));
    REQUIRE(items.size() == 7);
    REQUIRE(items.front().type() == typeid(std::nullptr_t));
    REQUIRE(std::any_cast<bool>(items.at(1)));
    REQUIRE(items.at(5).type() == typeid(Parser::Map));
    REQUIRE(items.at(6).type() == typeid(Parser::Sequence));
    REQUIRE(EncodeJson(map) == json);
    REQUIRE(map == DecodeJson(EncodeJson(map)));
    REQUIRE(Parser::Hash(map) == Parser::Hash(DecodeJson(json)));
}

TEST_CASE("JST_SERDES round-trips typed configurations through JSON", "[core][serialization][json][serdes]") {
    const OuterConfig outer{{42, true}, "nested"};
    const SequenceConfig steps{{{2, false}, {5, true}}};
    const PrimitiveVectorConfig vectors{{1, 2, 3}, {1.5f, 2.5f}, {3.25, 4.75}, {"a", "b"}};
    const NestedVectorConfig matrix{{{1, 2}, {}, {3}}};
    const std::vector<bool> flags{true, false};

    Parser::Map map;
    REQUIRE(Parser::Serialize(map, "outer", outer) == Result::SUCCESS);
    REQUIRE(Parser::Serialize(map, "steps", steps) == Result::SUCCESS);
    REQUIRE(Parser::Serialize(map, "vectors", vectors) == Result::SUCCESS);
    REQUIRE(Parser::Serialize(map, "matrix", matrix) == Result::SUCCESS);
    REQUIRE(Parser::Serialize(map, "flags", flags) == Result::SUCCESS);
    const auto json = EncodeJson(map);
    REQUIRE(json["outer"]["inner"]["enabled"].is_boolean());
    REQUIRE(json["vectors"]["counts"].is_array());
    REQUIRE(json["vectors"]["ratios"][0].is_number_float());
    REQUIRE(json["matrix"]["groups"][1] == nlohmann::json::array());
    const auto decoded = DecodeJson(json);

    OuterConfig restoredOuter;
    SequenceConfig restoredSteps;
    PrimitiveVectorConfig restoredVectors;
    NestedVectorConfig restoredMatrix;
    std::vector<bool> restoredFlags;
    REQUIRE(Parser::Deserialize(decoded, "outer", restoredOuter) == Result::SUCCESS);
    REQUIRE(Parser::Deserialize(decoded, "steps", restoredSteps) == Result::SUCCESS);
    REQUIRE(Parser::Deserialize(decoded, "vectors", restoredVectors) == Result::SUCCESS);
    REQUIRE(Parser::Deserialize(decoded, "matrix", restoredMatrix) == Result::SUCCESS);
    REQUIRE(Parser::Deserialize(decoded, "flags", restoredFlags) == Result::SUCCESS);
    REQUIRE(restoredOuter.hash() == outer.hash());
    REQUIRE(restoredSteps.hash() == steps.hash());
    REQUIRE(restoredVectors.hash() == vectors.hash());
    REQUIRE(restoredMatrix.groups == matrix.groups);
    REQUIRE(restoredFlags == flags);

    SECTION("domain values reuse existing Parser string encodings") {
        const Parser::Map domains{
            {"device", DeviceType::CPU}, {"runtime", RuntimeType::NATIVE},
            {"gain", CF32{1.5f, -2.5f}}, {"taps", std::vector<CF64>{{1.0, 2.0}, {3.0, -4.0}}},
            {"range", Range<F32>{-1.0f, 1.0f}}, {"size", Extent2D<U64>{640, 480}},
        };
        const auto encoded = EncodeJson(domains);
        for (const auto& [key, value] : domains) {
            std::string text;
            REQUIRE(Parser::TypedToString(value, text) == Result::SUCCESS);
            REQUIRE(encoded.at(key) == text);
        }
        const auto restored = DecodeJson(encoded);
        CF32 gain;
        std::vector<CF64> taps;
        REQUIRE(Parser::Deserialize(restored, "gain", gain) == Result::SUCCESS);
        REQUIRE(Parser::Deserialize(restored, "taps", taps) == Result::SUCCESS);
        REQUIRE(gain == std::any_cast<CF32>(domains.at("gain")));
        REQUIRE(taps == std::any_cast<std::vector<CF64>>(domains.at("taps")));
    }
}

TEST_CASE("Parser JSON distinguishes null from omitted optional fields", "[core][serialization][json]") {
    OptionalConfig config;
    Parser::Map encoded;
    REQUIRE(config.serialize(encoded) == Result::SUCCESS);
    REQUIRE(EncodeJson(encoded) == nlohmann::json::object());

    config.label = "present";
    config.steps = std::vector<U64>{1, 2};
    const auto nulls = DecodeJson(nlohmann::json{{"label", nullptr}, {"steps", nullptr}});
    REQUIRE(config.deserialize(nulls) == Result::SUCCESS);
    REQUIRE_FALSE(config.label.has_value());
    REQUIRE_FALSE(config.steps.has_value());
    REQUIRE(EncodeJson(nulls).at("label").is_null());

    std::string required = "keep";
    REQUIRE(Parser::Deserialize(nulls, "label", required) == Result::ERROR);
    REQUIRE(required == "keep");
}

TEST_CASE("Parser JSON values use checked numeric deserialization", "[core][serialization][json]") {
    const auto map = DecodeJson(nlohmann::json::parse(R"({
        "small": 255, "large": 256, "negative": -1, "fraction": 1.5,
        "unsigned": 18446744073709551615, "float_overflow": 1e100
    })"));
    U8 small = 0;
    REQUIRE(Parser::Deserialize(map, "small", small) == Result::SUCCESS);
    REQUIRE(small == 255);
    for (const auto* key : {"large", "negative", "fraction"}) {
        REQUIRE(Parser::Deserialize(map, key, small) == Result::ERROR);
        REQUIRE(small == 255);
    }
    I64 signedValue = 7;
    REQUIRE(Parser::Deserialize(map, "unsigned", signedValue) == Result::ERROR);
    REQUIRE(signedValue == 7);
    F32 floatValue = 2.0f;
    REQUIRE(Parser::Deserialize(map, "float_overflow", floatValue) == Result::ERROR);
    REQUIRE(floatValue == 2.0f);
}

TEST_CASE("Parser JSON adapters reject unsupported values", "[core][serialization][json]") {
    REQUIRE(DecodeStatus(nlohmann::json::array()) == Result::ERROR);
    REQUIRE(DecodeStatus(nullptr) == Result::ERROR);
    REQUIRE(DecodeStatus("text") == Result::ERROR);
    REQUIRE(DecodeStatus(nlohmann::json::parse("{", nullptr, false)) == Result::ERROR);
    REQUIRE(DecodeStatus(nlohmann::json{{"bytes", nlohmann::json::binary({1, 2})}}) == Result::ERROR);
    REQUIRE(EncodeStatus(std::any{UnsupportedValue{}}) == Result::ERROR);
    for (const auto value : {std::numeric_limits<F64>::infinity(), std::numeric_limits<F64>::quiet_NaN()}) {
        REQUIRE(EncodeStatus(Parser::Map{{"number", value}}) == Result::ERROR);
        REQUIRE(EncodeStatus(Parser::Map{{"array", std::vector<F64>{1.0, value}}}) == Result::ERROR);
        REQUIRE(DecodeStatus(nlohmann::json{{"number", value}}) == Result::ERROR);
    }
}
