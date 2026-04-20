#include <catch2/catch_test_macros.hpp>

#include "common.hh"

using namespace Jetstream;
using namespace TestParser;

TEST_CASE("Parser YAML round-trips mixed maps and sequences", "[parser][yaml]") {
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

TEST_CASE("Parser::YamlDecode handles empty and quoted input", "[parser][yaml]") {
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

TEST_CASE("Parser YAML preserves multiline scalars", "[parser][yaml]") {
    Parser::Map source;
    source["note"] = std::string("first line\nsecond line\n");

    std::string yaml;
    REQUIRE(Parser::YamlEncode(source, yaml) == Result::SUCCESS);
    REQUIRE(yaml.find('|') != std::string::npos);

    Parser::Map restored;
    REQUIRE(Parser::YamlDecode(yaml, restored) == Result::SUCCESS);
    REQUIRE(std::any_cast<std::string>(restored.at("note")) == "first line\nsecond line\n");
}

TEST_CASE("Parser YAML skips empty keys during encoding", "[parser][yaml]") {
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

TEST_CASE("Parser::YamlDecode rejects invalid YAML", "[parser][yaml]") {
    Parser::Map data;
    REQUIRE(Parser::YamlDecode("label: [1, 2\n", data) == Result::ERROR);
}
