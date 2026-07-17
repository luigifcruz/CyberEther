#include <catch2/catch_test_macros.hpp>

#include "fixture.hh"

using namespace Jetstream;
using namespace TestSerialization;

TEST_CASE("Parser YAML round-trips mixed maps and sequences", "[core][serialization][yaml]") {
    Parser::Map source;
    source["label"] = std::string("graph");
    source["inner"] = MakeStringInnerMap(7, true);

    Parser::Sequence steps;
    steps.push_back(std::string("prepare"));
    steps.push_back(MakeStringInnerMap(11, false));
    source["steps"] = steps;

    std::string yaml;
    REQUIRE(Parser::YamlEncode(source, yaml) == Result::SUCCESS);

    Parser::Map restored;
    REQUIRE(Parser::YamlDecode(yaml, restored) == Result::SUCCESS);
    REQUIRE(restored.contains("label"));
    REQUIRE(std::any_cast<std::string>(restored.at("label")) == "graph");

    REQUIRE(restored.contains("inner"));
    REQUIRE(restored.at("inner").type() == typeid(Parser::Map));

    const auto& inner = std::any_cast<const Parser::Map&>(restored.at("inner"));
    REQUIRE(std::any_cast<std::string>(inner.at("gain")) == "7");
    REQUIRE(std::any_cast<std::string>(inner.at("enabled")) == "true");

    REQUIRE(restored.contains("steps"));
    REQUIRE(restored.at("steps").type() == typeid(Parser::Sequence));

    const auto& decodedSteps = std::any_cast<const Parser::Sequence&>(restored.at("steps"));
    REQUIRE(decodedSteps.size() == 2);
    REQUIRE(std::any_cast<std::string>(decodedSteps.at(0)) == "prepare");
    REQUIRE(decodedSteps.at(1).type() == typeid(Parser::Map));
}

TEST_CASE("Parser::YamlDecode handles empty and quoted input", "[core][serialization][yaml]") {
    SECTION("empty documents clear the destination map") {
        Parser::Map data;
        data["label"] = std::string("present");

        REQUIRE(Parser::YamlDecode("", data) == Result::SUCCESS);
        REQUIRE(data.empty());
    }

    SECTION("quoted scalars are normalized") {
        Parser::Map data;
        REQUIRE(Parser::YamlDecode("label: 'quoted value'\n", data) == Result::SUCCESS);
        REQUIRE(std::any_cast<std::string>(data.at("label")) == "quoted value");
    }
}

TEST_CASE("Parser YAML preserves multiline scalars", "[core][serialization][yaml]") {
    Parser::Map source;
    source["note"] = std::string("first line\nsecond line\n");

    std::string yaml;
    REQUIRE(Parser::YamlEncode(source, yaml) == Result::SUCCESS);
    REQUIRE(yaml.find('|') != std::string::npos);

    Parser::Map restored;
    REQUIRE(Parser::YamlDecode(yaml, restored) == Result::SUCCESS);
    REQUIRE(std::any_cast<std::string>(restored.at("note")) == "first line\nsecond line\n");
}

TEST_CASE("Parser YAML round-trips empty containers and scalar syntax", "[core][serialization][yaml]") {
    Parser::Map source;
    source["empty-map"] = Parser::Map{};
    source["empty-sequence"] = Parser::Sequence{};
    source["boolean-like"] = std::string("true");
    source["punctuation"] = std::string("value: # [x]");
    source["spaces"] = std::string("  padded  ");

    std::string yaml;
    REQUIRE(Parser::YamlEncode(source, yaml) == Result::SUCCESS);

    Parser::Map restored;
    REQUIRE(Parser::YamlDecode(yaml, restored) == Result::SUCCESS);
    REQUIRE(std::any_cast<const Parser::Map&>(restored.at("empty-map")).empty());
    REQUIRE(std::any_cast<const Parser::Sequence&>(restored.at("empty-sequence")).empty());
    REQUIRE(std::any_cast<std::string>(restored.at("boolean-like")) == "true");
    REQUIRE(std::any_cast<std::string>(restored.at("punctuation")) == "value: # [x]");
    REQUIRE(std::any_cast<std::string>(restored.at("spaces")) == "  padded  ");
}

TEST_CASE("JST_SERDES round-trips through YAML string storage", "[core][serialization][yaml][serdes]") {
    OuterConfig source;
    source.inner.gain = 23;
    source.inner.enabled = true;
    source.label = "yaml round-trip";

    Parser::Map encoded;
    REQUIRE(source.serialize(encoded) == Result::SUCCESS);

    std::string yaml;
    REQUIRE(Parser::YamlEncode(encoded, yaml) == Result::SUCCESS);

    Parser::Map decoded;
    REQUIRE(Parser::YamlDecode(yaml, decoded) == Result::SUCCESS);

    OuterConfig restored;
    REQUIRE(restored.deserialize(decoded) == Result::SUCCESS);
    REQUIRE(restored.inner.gain == source.inner.gain);
    REQUIRE(restored.inner.enabled == source.inner.enabled);
    REQUIRE(restored.label == source.label);
}

TEST_CASE("Parser YAML skips empty keys during encoding", "[core][serialization][yaml]") {
    Parser::Map source;
    source[""] = std::string("skip");
    source["label"] = std::string("keep");

    std::string yaml;
    REQUIRE(Parser::YamlEncode(source, yaml) == Result::SUCCESS);
    REQUIRE(yaml.find("skip") == std::string::npos);

    Parser::Map restored;
    REQUIRE(Parser::YamlDecode(yaml, restored) == Result::SUCCESS);
    REQUIRE(!restored.contains(""));
    REQUIRE(std::any_cast<std::string>(restored.at("label")) == "keep");
}

TEST_CASE("Parser::YamlDecode rejects invalid YAML", "[core][serialization][yaml]") {
    Parser::Map data;
    data["label"] = std::string("unchanged");

    REQUIRE(Parser::YamlDecode("label: [1, 2\n", data) == Result::ERROR);
    REQUIRE(data.size() == 1);
    REQUIRE(std::any_cast<std::string>(data.at("label")) == "unchanged");
}

TEST_CASE("Parser YAML rejects non-map document roots", "[core][serialization][yaml][errors]") {
    SECTION("scalar roots") {
        Parser::Map data;
        data["label"] = std::string("unchanged");

        REQUIRE(Parser::YamlDecode("scalar\n", data) == Result::ERROR);
        REQUIRE(std::any_cast<std::string>(data.at("label")) == "unchanged");
        REQUIRE(Parser::YamlDecode("---\nscalar\n", data) == Result::ERROR);
        REQUIRE(std::any_cast<std::string>(data.at("label")) == "unchanged");
    }

    SECTION("sequence roots") {
        Parser::Map data;
        data["label"] = std::string("unchanged");

        REQUIRE(Parser::YamlDecode("- label: first\n- label: second\n", data) == Result::ERROR);
        REQUIRE(std::any_cast<std::string>(data.at("label")) == "unchanged");
        REQUIRE(Parser::YamlDecode("---\n- label: first\n- label: second\n", data) ==
                Result::ERROR);
        REQUIRE(std::any_cast<std::string>(data.at("label")) == "unchanged");
    }

    SECTION("null roots") {
        Parser::Map data;
        data["label"] = std::string("unchanged");

        REQUIRE(Parser::YamlDecode("---\n", data) == Result::ERROR);
        REQUIRE(std::any_cast<std::string>(data.at("label")) == "unchanged");
    }

    SECTION("explicit map roots") {
        Parser::Map data;
        data["label"] = std::string("unchanged");

        REQUIRE(Parser::YamlDecode("---\nlabel: decoded\n", data) == Result::SUCCESS);
        REQUIRE(std::any_cast<std::string>(data.at("label")) == "decoded");
    }
}

TEST_CASE("Parser::YamlEncode preserves output on unsupported values", "[core][serialization][yaml][errors]") {
    Parser::Map data;
    data["unsupported"] = UnsupportedValue{};

    std::string yaml = "unchanged";
    REQUIRE(Parser::YamlEncode(data, yaml) == Result::ERROR);
    REQUIRE(yaml == "unchanged");
}
