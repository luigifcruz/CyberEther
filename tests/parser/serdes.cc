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

int main(int argc, char* argv[]) {
    JST_LOG_SET_DEBUG_LEVEL(0);
    return Catch::Session().run(argc, argv);
}
