#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "common.hh"

using namespace Jetstream;
using namespace TestParser;

namespace {

void RequireF32VectorEq(const std::vector<F32>& actual, const std::vector<F32>& expected) {
    REQUIRE(actual.size() == expected.size());

    for (U64 i = 0; i < actual.size(); ++i) {
        REQUIRE(actual[i] == Catch::Approx(expected[i]));
    }
}

void RequireF64VectorEq(const std::vector<F64>& actual, const std::vector<F64>& expected) {
    REQUIRE(actual.size() == expected.size());

    for (U64 i = 0; i < actual.size(); ++i) {
        REQUIRE(actual[i] == Catch::Approx(expected[i]));
    }
}

}  // namespace

TEST_CASE("JST_SERDES serializes nested structs as maps", "[parser][serdes]") {
    OuterConfig source;
    source.inner.gain = 42;
    source.inner.enabled = true;
    source.label = "nested";

    Parser::Map data;
    REQUIRE(source.serialize(data) == Result::SUCCESS);
    REQUIRE(data.contains("inner"));
    REQUIRE(data.at("inner").type() == typeid(Parser::Map));

    const auto& nested = std::any_cast<const Parser::Map&>(data.at("inner"));
    REQUIRE(nested.contains("gain"));
    REQUIRE(nested.contains("enabled"));
    REQUIRE(std::any_cast<U64>(nested.at("gain")) == 42);
    REQUIRE(std::any_cast<bool>(nested.at("enabled")));

    OuterConfig restored;
    REQUIRE(restored.deserialize(data) == Result::SUCCESS);
    REQUIRE(restored.inner.gain == 42);
    REQUIRE(restored.inner.enabled);
    REQUIRE(restored.label == "nested");
}

TEST_CASE("JST_SERDES cascades string-backed nested maps", "[parser][serdes]") {
    Parser::Map data;
    data["inner"] = MakeStringInnerMap(7, true);
    data["label"] = std::string("from-yaml");

    OuterConfig restored;
    REQUIRE(restored.deserialize(data) == Result::SUCCESS);
    REQUIRE(restored.inner.gain == 7);
    REQUIRE(restored.inner.enabled);
    REQUIRE(restored.label == "from-yaml");
}

TEST_CASE("JST_SERDES serializes vectors of nested structs as sequences", "[parser][serdes]") {
    SequenceConfig source;
    source.steps.push_back({.gain = 2, .enabled = true});
    source.steps.push_back({.gain = 5, .enabled = false});

    Parser::Map data;
    REQUIRE(source.serialize(data) == Result::SUCCESS);
    REQUIRE(data.contains("steps"));
    REQUIRE(data.at("steps").type() == typeid(Parser::Sequence));

    const auto& steps = std::any_cast<const Parser::Sequence&>(data.at("steps"));
    REQUIRE(steps.size() == 2);
    REQUIRE(steps.at(0).type() == typeid(Parser::Map));

    const auto& first = std::any_cast<const Parser::Map&>(steps.at(0));
    REQUIRE(std::any_cast<U64>(first.at("gain")) == 2);
    REQUIRE(std::any_cast<bool>(first.at("enabled")));

    SequenceConfig restored;
    REQUIRE(restored.deserialize(data) == Result::SUCCESS);
    REQUIRE(restored.steps.size() == 2);
    REQUIRE(restored.steps.at(0).gain == 2);
    REQUIRE(restored.steps.at(0).enabled);
    REQUIRE(restored.steps.at(1).gain == 5);
    REQUIRE(!restored.steps.at(1).enabled);
}

TEST_CASE("JST_SERDES deserializes string-backed sequences of nested structs", "[parser][serdes]") {
    Parser::Sequence steps;
    steps.push_back(MakeStringInnerMap(13, true));
    steps.push_back(MakeStringInnerMap(21, false));

    Parser::Map data;
    data["steps"] = steps;

    SequenceConfig restored;
    REQUIRE(restored.deserialize(data) == Result::SUCCESS);
    REQUIRE(restored.steps.size() == 2);
    REQUIRE(restored.steps.at(0).gain == 13);
    REQUIRE(restored.steps.at(0).enabled);
    REQUIRE(restored.steps.at(1).gain == 21);
    REQUIRE(!restored.steps.at(1).enabled);
}

TEST_CASE("JST_SERDES omits null optional fields during serialization", "[parser][serdes]") {
    OptionalConfig source;

    Parser::Map data;
    REQUIRE(source.serialize(data) == Result::SUCCESS);
    REQUIRE(!data.contains("label"));
    REQUIRE(!data.contains("steps"));
}

TEST_CASE("JST_SERDES resets missing optional fields during deserialization", "[parser][serdes]") {
    OptionalConfig restored;
    restored.label = "present";
    restored.steps = std::vector<U64>{1, 2, 3};

    Parser::Map data;
    REQUIRE(restored.deserialize(data) == Result::SUCCESS);
    REQUIRE(!restored.label.has_value());
    REQUIRE(!restored.steps.has_value());
}

TEST_CASE("JST_SERDES round-trips present optional fields", "[parser][serdes]") {
    OptionalConfig source;
    source.label = "optional";
    source.steps = std::vector<U64>{4, 8, 15};

    Parser::Map data;
    REQUIRE(source.serialize(data) == Result::SUCCESS);
    REQUIRE(data.contains("label"));
    REQUIRE(data.contains("steps"));

    OptionalConfig restored;
    REQUIRE(restored.deserialize(data) == Result::SUCCESS);
    REQUIRE(restored.label.has_value());
    REQUIRE(restored.label.value() == "optional");
    REQUIRE(restored.steps.has_value());
    REQUIRE(restored.steps.value() == std::vector<U64>({4, 8, 15}));
}

TEST_CASE("Parser stores primitive vectors directly and nested vectors as sequences", "[parser][serdes]") {
    SECTION("primitive vectors are stored directly") {
        PrimitiveVectorConfig source;
        source.counts = {1, 2, 3};
        source.ratios = {1.5f, 2.5f};
        source.weights = {3.25, 4.75};

        Parser::Map data;
        REQUIRE(source.serialize(data) == Result::SUCCESS);
        REQUIRE(data.at("counts").type() == typeid(std::vector<U64>));
        REQUIRE(data.at("ratios").type() == typeid(std::vector<F32>));
        REQUIRE(data.at("weights").type() == typeid(std::vector<F64>));

        PrimitiveVectorConfig restored;
        REQUIRE(restored.deserialize(data) == Result::SUCCESS);
        REQUIRE(restored.counts == source.counts);
        RequireF32VectorEq(restored.ratios, source.ratios);
        RequireF64VectorEq(restored.weights, source.weights);
    }

    SECTION("string-backed primitive vectors deserialize through Parser::Deserialize") {
        Parser::Map data;
        data["counts"] = std::string("[1, 2, 3]");
        data["ratios"] = std::string("[1.5, 2.5]");
        data["weights"] = std::string("[3.25, 4.75]");

        PrimitiveVectorConfig restored;
        REQUIRE(restored.deserialize(data) == Result::SUCCESS);
        REQUIRE(restored.counts == std::vector<U64>({1, 2, 3}));
        RequireF32VectorEq(restored.ratios, std::vector<F32>{1.5f, 2.5f});
        RequireF64VectorEq(restored.weights, std::vector<F64>{3.25, 4.75});
    }

    SECTION("nested vectors are stored as Parser::Sequence") {
        NestedVectorConfig source;
        source.groups = {{1, 2}, {3, 5, 8}};

        Parser::Map data;
        REQUIRE(source.serialize(data) == Result::SUCCESS);
        REQUIRE(data.at("groups").type() == typeid(Parser::Sequence));

        const auto& groups = std::any_cast<const Parser::Sequence&>(data.at("groups"));
        REQUIRE(groups.size() == 2);
        REQUIRE(groups.at(0).type() == typeid(std::vector<U64>));
        REQUIRE(std::any_cast<const std::vector<U64>&>(groups.at(0)) == std::vector<U64>({1, 2}));

        NestedVectorConfig restored;
        REQUIRE(restored.deserialize(data) == Result::SUCCESS);
        REQUIRE(restored.groups == source.groups);
    }
}

TEST_CASE("Parser::Serialize overwrites and erases existing entries", "[parser][serdes]") {
    SECTION("overwrites an existing entry") {
        Parser::Map data;
        data["value"] = std::string("old");

        REQUIRE(Parser::Serialize(data, "value", U64{9}) == Result::SUCCESS);
        REQUIRE(data.contains("value"));
        REQUIRE(data.at("value").type() == typeid(U64));
        REQUIRE(std::any_cast<U64>(data.at("value")) == 9);
    }

    SECTION("erases an existing entry when serializing an empty optional") {
        Parser::Map data;
        data["value"] = std::string("old");

        const std::optional<std::string> value;
        REQUIRE(Parser::Serialize(data, "value", value) == Result::SUCCESS);
        REQUIRE(!data.contains("value"));
    }
}

TEST_CASE("Parser::Deserialize uses exact-type fast paths", "[parser][serdes]") {
    Parser::Map data;
    data["gain"] = U64{17};
    data["counts"] = std::vector<U64>{1, 2, 3};
    data["inner"] = MakeInnerMap(4, true);

    Parser::Sequence steps;
    steps.push_back(std::string("first"));
    steps.push_back(U64{2});
    data["steps"] = steps;

    U64 gain = 0;
    REQUIRE(Parser::Deserialize(data, "gain", gain) == Result::SUCCESS);
    REQUIRE(gain == 17);

    std::vector<U64> counts;
    REQUIRE(Parser::Deserialize(data, "counts", counts) == Result::SUCCESS);
    REQUIRE(counts == std::vector<U64>({1, 2, 3}));

    Parser::Map inner;
    REQUIRE(Parser::Deserialize(data, "inner", inner) == Result::SUCCESS);
    REQUIRE(inner.contains("gain"));
    REQUIRE(std::any_cast<U64>(inner.at("gain")) == 4);

    Parser::Sequence decodedSteps;
    REQUIRE(Parser::Deserialize(data, "steps", decodedSteps) == Result::SUCCESS);
    REQUIRE(decodedSteps.size() == 2);
    REQUIRE(std::any_cast<std::string>(decodedSteps.at(0)) == "first");
    REQUIRE(std::any_cast<U64>(decodedSteps.at(1)) == 2);
}

TEST_CASE("Parser::Deserialize handles missing and uninitialized values", "[parser][serdes]") {
    SECTION("leaves missing non-optional values unchanged") {
        Parser::Map data;
        U64 gain = 99;

        REQUIRE(Parser::Deserialize(data, "gain", gain) == Result::SUCCESS);
        REQUIRE(gain == 99);
    }

    SECTION("returns an error for a present but uninitialized entry") {
        Parser::Map data;
        data["gain"] = std::any{};

        U64 gain = 0;
        REQUIRE(Parser::Deserialize(data, "gain", gain) == Result::ERROR);
    }
}

TEST_CASE("Parser::Deserialize reports incompatible types", "[parser][serdes]") {
    Parser::Map data;

    SECTION("Parser::Map rejects strings") {
        data["value"] = std::string("wrong");
        Parser::Map decoded;
        REQUIRE(Parser::Deserialize(data, "value", decoded) == Result::ERROR);
    }

    SECTION("Parser::Map rejects scalars") {
        data["value"] = U64{7};
        Parser::Map decoded;
        REQUIRE(Parser::Deserialize(data, "value", decoded) == Result::ERROR);
    }

    SECTION("Parser::Sequence rejects non-sequences") {
        data["value"] = U64{7};
        Parser::Sequence decoded;
        REQUIRE(Parser::Deserialize(data, "value", decoded) == Result::ERROR);
    }

    SECTION("vectors of nested types reject strings") {
        data["steps"] = std::string("wrong");
        SequenceConfig decoded;
        REQUIRE(decoded.deserialize(data) == Result::ERROR);
    }

    SECTION("vectors of nested types reject non-sequences") {
        data["steps"] = U64{7};
        SequenceConfig decoded;
        REQUIRE(decoded.deserialize(data) == Result::ERROR);
    }

    SECTION("nested serdes types reject strings") {
        data["inner"] = std::string("wrong");
        OuterConfig decoded;
        REQUIRE(decoded.deserialize(data) == Result::ERROR);
    }

    SECTION("nested serdes types reject non-map values") {
        data["inner"] = Parser::Sequence{};
        OuterConfig decoded;
        REQUIRE(decoded.deserialize(data) == Result::ERROR);
    }

    SECTION("scalars reject non-string mismatches") {
        data["gain"] = bool{true};
        U64 gain = 0;
        REQUIRE(Parser::Deserialize(data, "gain", gain) == Result::ERROR);
    }
}

TEST_CASE("Parser::Serialize returns errors from throwing serialize methods", "[parser][serdes]") {
    Parser::Map data;
    REQUIRE(Parser::Serialize(data, "throwing", ThrowingConfig{}) == Result::ERROR);
}
