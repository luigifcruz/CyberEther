#include <catch2/catch_test_macros.hpp>

#include <jetstream/block_interface.hh>
#include <jetstream/parser.hh>

#include <limits>

using namespace Jetstream;

TEST_CASE("Sequence construction owns strings and preserves interface properties",
          "[core][serialization][interface]") {
    std::string extension = "iq";
    Parser::Map format{
        {"type", "filepicker"},
        {"extensions", Parser::MakeSequence({"bin", "raw", extension.c_str()})},
    };
    extension.assign("changed");
    REQUIRE(Block::Interface::ValidateFormat(format) == Result::SUCCESS);
    std::vector<std::string> extensions;
    REQUIRE(Parser::Deserialize(format, "extensions", extensions) == Result::SUCCESS);
    REQUIRE(extensions == std::vector<std::string>{"bin", "raw", "iq"});
    std::string yaml;
    REQUIRE(Parser::YamlEncode(format, yaml) == Result::SUCCESS);
    Parser::Map restored;
    REQUIRE(Parser::YamlDecode(yaml, restored) == Result::SUCCESS);
    REQUIRE(restored == format);

    const auto mixed = Parser::MakeSequence({"", U64{7}, true, Parser::Map{{"key", "value"}},
                                            Parser::MakeSequence({"nested"})});
    REQUIRE(std::any_cast<std::string>(mixed[0]).empty());
    REQUIRE(std::any_cast<U64>(mixed[1]) == 7);
    REQUIRE(std::any_cast<bool>(mixed[2]));
    REQUIRE(std::any_cast<Parser::Map>(mixed[3]) == Parser::Map{{"key", "value"}});
    REQUIRE(Parser::Equal(mixed[4], Parser::Sequence{std::string("nested")}));
    REQUIRE(Parser::MakeSequence({}).empty());
}

TEST_CASE("Parser maps own initializer strings and preserve nested dropdown data",
          "[core][serialization][interface]") {
    std::string label = "Radio, (USB): \"α\"";
    const std::string arguments = "driver=lime,serial=00123,remote=tcp://host:55132";
    Parser::Map format{
        {"type", "dropdown"},
        {"options", Parser::Sequence{
            Parser::Map{{"label", "No device selected"}, {"value", ""}},
            Parser::Map{{"label", label.c_str()}, {"value", arguments}},
        }},
    };
    label.assign("changed");
    REQUIRE(Block::Interface::ValidateFormat(format) == Result::SUCCESS);
    std::string yaml;
    REQUIRE(Parser::YamlEncode(format, yaml) == Result::SUCCESS);
    Parser::Map decoded;
    REQUIRE(Parser::YamlDecode(yaml, decoded) == Result::SUCCESS);
    REQUIRE(decoded == format);
    REQUIRE(Block::Interface::ValidateFormat(decoded) == Result::SUCCESS);
    const auto options = Parser::Get<std::vector<Parser::Map>>(decoded, "options");
    REQUIRE(options.size() == 2);
    REQUIRE(Parser::Get<std::string>(options[0], "value").empty());
    REQUIRE(Parser::Get<std::string>(options[1], "label") == "Radio, (USB): \"α\"");
    REQUIRE(Parser::Get<std::string>(options[1], "value") == arguments);
    auto copy = decoded;
    copy["type"] = std::string("text");
    REQUIRE(copy != decoded);
    REQUIRE(Parser::Get<std::string>(decoded, "type") == "dropdown");
}

TEST_CASE("Interface descriptors validate every control and internal metric",
          "[core][serialization][interface]") {
    for (const auto* type : {"bool", "int", "uint", "float", "range", "text", "multiline", "vector",
                             "vector-inline", "filepicker", "filesave", "markdown", "python"}) {
        CAPTURE(type);
        REQUIRE(Block::Interface::ValidateFormat({{"type", type}}) == Result::SUCCESS);
    }
    REQUIRE(Block::Interface::ValidateFormat({{"type", "tensor-config"}, {"index", 2}, {"source", "outputs"}}) == Result::SUCCESS);
    for (const auto* type : {"label", "progressbar", "table"}) {
        REQUIRE(Block::Interface::ValidateFormat({{"type", type}}, true) == Result::SUCCESS);
        REQUIRE(Block::Interface::ValidateFormat({{"type", type}, {"visibility", "public"}}, true) == Result::SUCCESS);
        REQUIRE(Block::Interface::ValidateFormat({{"type", type}, {"visibility", "internal"}}, true) == Result::SUCCESS);
    }
    REQUIRE(Block::Interface::ValidateFormat({{"type", "vector"}, {"value_type", "float"}}) == Result::SUCCESS);
    REQUIRE(Block::Interface::ValidateFormat({{"type", "vector"}, {"value_type", "uint"}}) == Result::SUCCESS);
    REQUIRE(Block::Interface::ValidateFormat({{"type", "timing"}, {"visibility", "internal"}}, true) == Result::SUCCESS);
    REQUIRE(Block::Interface::ValidateFormat({{"type", "python-diagnostic"}, {"visibility", "internal"}}, true) == Result::SUCCESS);
    REQUIRE(Block::Interface::ValidateFormat({{"type", "float"}, {"unit", "arbitrary"}, {"scale", 42.0},
                                             {"precision", 3}, {"step_config", "step"}}) == Result::SUCCESS);
}

TEST_CASE("Malformed interface descriptors fail without throwing",
          "[core][serialization][interface]") {
    const std::vector<Parser::Map> formats{
        {}, {{"type", ""}}, {{"type", "unknown"}}, {{"type", 42}},
        {{"type", "float"}, {"scale", 0}},
        {{"type", "float"}, {"scale", std::numeric_limits<F64>::infinity()}},
        {{"type", "float"}, {"precision", "invalid"}},
        {{"type", "float"}, {"precision", -1}},
        {{"type", "float"}, {"precision", 2.5}},
        {{"type", "range"}, {"min", 2}, {"max", 1}},
        {{"type", "range"}, {"min", -1}, {"value_type", "uint"}},
        {{"type", "vector"}, {"value_type", U64{1}}},
        {{"type", "vector"}, {"value_type", "invalid"}},
        {{"type", "tensor-config"}, {"index", -1}, {"source", "outputs"}},
        {{"type", "tensor-config"}},
        {{"type", "multiline"}, {"collapsible", 2}},
        {{"type", "filepicker"}, {"extensions", "iq,raw"}},
        {{"type", "dropdown"}},
        {{"type", "dropdown"}, {"options", Parser::Sequence{Parser::Map{{"label", "Missing value"}}}}},
        {{"type", "dropdown"}, {"options", Parser::Sequence{std::string("not an option")}}},
        {{"type", "dropdown"}, {"options", Parser::Sequence{Parser::Map{{"label", "Number"}, {"value", U64{1}}}}}},
        {{"type", "dropdown"}, {"options", Parser::Sequence{Parser::Map{{"label", "Map"}, {"value", Parser::Map{}}}}}},
        {{"type", "bool"}, {"visibility", "hidden"}},
        {{"type", "bool"}, {"visibility", "public"}},
        {{"type", "bool"}, {"visibility", "internal"}},
    };
    for (const auto& format : formats) {
        REQUIRE(Block::Interface::ValidateFormat(format) == Result::ERROR);
    }
}

TEST_CASE("Typed config decoding preserves integers and rejects narrowing overflow",
          "[core][serialization][interface]") {
    const U64 maximum = std::numeric_limits<U64>::max();
    U64 value = 0;
    REQUIRE(Parser::Deserialize({{"value", maximum}}, "value", value) == Result::SUCCESS);
    REQUIRE(value == maximum);
    REQUIRE(Parser::Deserialize({{"value", I64{-1}}}, "value", value) == Result::ERROR);
    REQUIRE(value == maximum);
    REQUIRE(Parser::Deserialize({{"value", 2.5}}, "value", value) == Result::ERROR);
    REQUIRE(Parser::Deserialize({{"value", std::ldexp(1.0, 64)}}, "value", value) == Result::ERROR);
    I64 signedValue = 0;
    REQUIRE(Parser::Deserialize({{"value", maximum}}, "value", signedValue) == Result::ERROR);
    const U64 signedMaximum = static_cast<U64>(std::numeric_limits<I64>::max());
    REQUIRE(Parser::Deserialize({{"value", signedMaximum}}, "value", signedValue) == Result::SUCCESS);
    REQUIRE(signedValue == std::numeric_limits<I64>::max());
    REQUIRE(Parser::Deserialize({{"value", signedMaximum + 1}}, "value", signedValue) == Result::ERROR);
    REQUIRE(signedValue == std::numeric_limits<I64>::max());
    F32 floatValue = 0;
    REQUIRE(Parser::Deserialize({{"value", U64{42}}}, "value", floatValue) == Result::SUCCESS);
    REQUIRE(floatValue == 42.0f);
    REQUIRE_FALSE(Parser::Equal(std::string("1"), U64{1}));
    REQUIRE(Parser::Equal(Parser::Sequence{Parser::Map{{"value", maximum}}},
                          Parser::Sequence{Parser::Map{{"value", maximum}}}));
}

TEST_CASE("Boolean and numeric values do not cross-convert",
          "[core][serialization][interface]") {
    for (const auto& number : Parser::MakeSequence({I64{-1}, I64{0}, I64{1}, U64{1}, U64{2}, 0.0f, 1.0})) {
        bool flag = true;
        REQUIRE(Parser::Deserialize({{"value", number}}, "value", flag) == Result::ERROR);
        REQUIRE(flag);
    }
    I64 integer = -1;
    F32 floating = -1.0f;
    for (const bool flag : {false, true}) {
        REQUIRE(Parser::Deserialize({{"value", flag}}, "value", integer) == Result::ERROR);
        REQUIRE(integer == -1);
        REQUIRE(Parser::Deserialize({{"value", flag}}, "value", floating) == Result::ERROR);
        REQUIRE(floating == -1.0f);
    }
    bool flag = false;
    REQUIRE(Parser::Deserialize({{"value", true}}, "value", flag) == Result::SUCCESS);
    REQUIRE(flag);
    REQUIRE(Parser::Deserialize({{"value", "false"}}, "value", flag) == Result::SUCCESS);
    REQUIRE_FALSE(flag);
    Parser::Map document;
    REQUIRE(Parser::YamlDecode("value: true\n", document) == Result::SUCCESS);
    REQUIRE(Parser::Deserialize(document, "value", floating) == Result::ERROR);
    REQUIRE(floating == -1.0f);
    REQUIRE(Parser::Deserialize(document, "value", flag) == Result::SUCCESS);
    REQUIRE(flag);
    std::optional<F32> optional = -1.0f;
    REQUIRE(Parser::Deserialize({{"value", true}}, "value", optional) == Result::ERROR);
    REQUIRE(optional == -1.0f);
    std::vector<F32> sequence{7.0f};
    REQUIRE(Parser::Deserialize({{"values", Parser::MakeSequence({1.0f, true})}}, "values", sequence) == Result::ERROR);
    REQUIRE(sequence == std::vector<F32>{7.0f});
}

TEST_CASE("Interface descriptors reject payloads that cannot support change detection",
          "[core][serialization][interface]") {
    struct Unsupported { int value = 1; };
    Parser::Map format{{"type", "float"}, {"metadata", Unsupported{}}};
    REQUIRE(Block::Interface::ValidateFormat(format) == Result::ERROR);
    format["metadata"] = Parser::MakeSequence({Parser::Map{{"nested", Unsupported{}}}});
    REQUIRE(Block::Interface::ValidateFormat(format) == Result::ERROR);
    format["metadata"] = std::numeric_limits<F64>::quiet_NaN();
    REQUIRE(Block::Interface::ValidateFormat(format) == Result::ERROR);
    format["metadata"] = Parser::MakeSequence({Parser::Map{{"nested", "supported"}}});
    REQUIRE(Block::Interface::ValidateFormat(format) == Result::SUCCESS);
    REQUIRE(format == format);
}

TEST_CASE("Get uses the same decoding rules as Deserialize with explicit fallback",
          "[core][serialization][interface]") {
    const Parser::Map values{{"precision", I64{3}}, {"invalid", "abc"}, {"negative", -1}};
    I32 precision = 2;
    REQUIRE(Parser::Deserialize(values, "precision", precision) == Result::SUCCESS);
    REQUIRE(Parser::Get<I32>(values, "precision", 2) == precision);
    REQUIRE(Parser::Get<I32>(values, "missing", 2) == 2);
    REQUIRE(Parser::Get<I32>(values, "invalid", 2) == 2);
    REQUIRE(Parser::Get<U64>(values, "negative", 7) == 7);
    std::optional<I32> optional;
    REQUIRE(Parser::Deserialize(values, "precision", optional) == Result::SUCCESS);
    REQUIRE(optional == 3);
    std::vector<U64> sequence;
    REQUIRE(Parser::Deserialize({{"items", Parser::Sequence{I32{1}, I64{2}}}}, "items", sequence) == Result::SUCCESS);
    REQUIRE(sequence == std::vector<U64>{1, 2});
    REQUIRE(Parser::Deserialize({{"items", Parser::Sequence{I32{1}, I64{-2}}}}, "items", sequence) == Result::ERROR);
    REQUIRE(sequence == std::vector<U64>{1, 2});
}
