#include <any>

#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>

#include "jetstream/parser.hh"

using namespace Jetstream;

namespace TestParserSerdes {

struct InnerConfig {
    U64 gain = 0;
    bool enabled = false;

    JST_SERDES(gain, enabled);
};

struct OuterConfig {
    InnerConfig inner;
    std::string label;

    JST_SERDES(inner, label);
};

struct MapConfig {
    std::unordered_map<std::string, InnerConfig> presets;

    JST_SERDES(presets);
};

}  // namespace TestParserSerdes

using namespace TestParserSerdes;

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
    Parser::Map nested;
    nested["gain"] = std::string("7");
    nested["enabled"] = std::string("true");

    Parser::Map data;
    data["inner"] = nested;
    data["label"] = std::string("from-yaml");

    OuterConfig restored;
    REQUIRE(restored.deserialize(data) == Result::SUCCESS);
    REQUIRE(restored.inner.gain == 7);
    REQUIRE(restored.inner.enabled);
    REQUIRE(restored.label == "from-yaml");
}

TEST_CASE("JST_SERDES hash includes nested fields", "[parser][hash]") {
    OuterConfig lhs;
    OuterConfig rhs;

    REQUIRE(lhs.hash() == rhs.hash());

    rhs.inner.gain = 9;
    REQUIRE(lhs.hash() != rhs.hash());
}

TEST_CASE("JST_SERDES serializes unordered maps of nested structs", "[parser][serdes]") {
    MapConfig source;
    source.presets["alpha"] = {.gain = 3, .enabled = true};
    source.presets["beta"] = {.gain = 8, .enabled = false};

    Parser::Map data;
    REQUIRE(source.serialize(data) == Result::SUCCESS);
    REQUIRE(data.contains("presets"));
    REQUIRE(data.at("presets").type() == typeid(Parser::Map));

    const auto& presets = std::any_cast<const Parser::Map&>(data.at("presets"));
    REQUIRE(presets.contains("alpha"));
    REQUIRE(presets.contains("beta"));
    REQUIRE(presets.at("alpha").type() == typeid(Parser::Map));

    const auto& alpha = std::any_cast<const Parser::Map&>(presets.at("alpha"));
    REQUIRE(std::any_cast<U64>(alpha.at("gain")) == 3);
    REQUIRE(std::any_cast<bool>(alpha.at("enabled")));

    MapConfig restored;
    REQUIRE(restored.deserialize(data) == Result::SUCCESS);
    REQUIRE(restored.presets.size() == 2);
    REQUIRE(restored.presets.at("alpha").gain == 3);
    REQUIRE(restored.presets.at("alpha").enabled);
    REQUIRE(restored.presets.at("beta").gain == 8);
    REQUIRE(!restored.presets.at("beta").enabled);
}

TEST_CASE("JST_SERDES deserializes string-backed unordered maps of nested structs", "[parser][serdes]") {
    Parser::Map alpha;
    alpha["gain"] = std::string("11");
    alpha["enabled"] = std::string("true");

    Parser::Map beta;
    beta["gain"] = std::string("5");
    beta["enabled"] = std::string("false");

    Parser::Map presets;
    presets["alpha"] = alpha;
    presets["beta"] = beta;

    Parser::Map data;
    data["presets"] = presets;

    MapConfig restored;
    REQUIRE(restored.deserialize(data) == Result::SUCCESS);
    REQUIRE(restored.presets.size() == 2);
    REQUIRE(restored.presets.at("alpha").gain == 11);
    REQUIRE(restored.presets.at("alpha").enabled);
    REQUIRE(restored.presets.at("beta").gain == 5);
    REQUIRE(!restored.presets.at("beta").enabled);
}

TEST_CASE("JST_SERDES hash for unordered maps is insertion-order independent", "[parser][hash]") {
    MapConfig lhs;
    lhs.presets["alpha"] = {.gain = 1, .enabled = true};
    lhs.presets["beta"] = {.gain = 2, .enabled = false};

    MapConfig rhs;
    rhs.presets["beta"] = {.gain = 2, .enabled = false};
    rhs.presets["alpha"] = {.gain = 1, .enabled = true};

    REQUIRE(lhs.hash() == rhs.hash());

    rhs.presets.at("alpha").gain = 9;
    REQUIRE(lhs.hash() != rhs.hash());
}

int main(int argc, char* argv[]) {
    JST_LOG_SET_DEBUG_LEVEL(0);
    return Catch::Session().run(argc, argv);
}
