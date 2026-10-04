#include <catch2/catch_test_macros.hpp>

#include <string>
#include <vector>

#include "compositor/default/live_markdown.hh"
#include "jetstream/flowgraph_environment.hh"

using namespace Jetstream;

TEST_CASE("Live markdown expands stats fences with LF and CRLF line endings",
          "[core][compositor][live-markdown]") {
    Flowgraph flowgraph;
    REQUIRE(flowgraph.environment().set("telemetry", Parser::Map{{"value", U64{42}}}) == Result::SUCCESS);

    for (const std::string ending : {"\n", "\r\n"}) {
        for (const std::string tag : {"stats", "stats \t", "stats{columns=2}", "stats {columns=2}"}) {
            CAPTURE(ending, tag);
            const std::string fence = "```" + tag + ending;
            const LiveMarkdown markdown(fence + "Value | ${env.telemetry.value}" + ending + "```" + ending);
            CHECK(markdown.dynamic());
            CHECK(markdown.expand(flowgraph.environment()) == fence + "Value | 42" + ending + "```" + ending);
        }
    }

    for (const std::string tag : {"statsx", "cpp"}) {
        CAPTURE(tag);
        const std::string source = "```" + tag + "\r\n${env.telemetry.value}\r\n```\r\n";
        const LiveMarkdown markdown(source);
        CHECK_FALSE(markdown.dynamic());
        CHECK(markdown.expand(flowgraph.environment()) == source);
    }
}

TEST_CASE("Live markdown preserves placeholders inside matching backtick runs",
          "[core][compositor][live-markdown]") {
    Flowgraph flowgraph;
    REQUIRE(flowgraph.environment().set("telemetry", Parser::Map{{"value", U64{42}}}) == Result::SUCCESS);

    for (const std::string span : {
        "`${env.telemetry.value}`",
        "``${env.telemetry.value}``",
        "```${env.telemetry.value}```",
        "``before ` ${env.telemetry.value} ` after``",
        "``before ``` ${env.telemetry.value} ``` after``",
        "`before `` ${env.telemetry.value} `` after`",
        "``${env.telemetry.value}\\``",
    }) {
        CAPTURE(span);
        const std::string source = "Code: " + span;
        const LiveMarkdown literal(source);
        CHECK_FALSE(literal.dynamic());
        CHECK(literal.expand(flowgraph.environment()) == source);

        const LiveMarkdown mixed("Before: ${env.telemetry.value}; " + span + "; after: ${env.telemetry.value}");
        CHECK(mixed.dynamic());
        CHECK(mixed.expand(flowgraph.environment()) == "Before: 42; " + span + "; after: 42");
    }
}

TEST_CASE("Live markdown expands placeholders after unmatched or escaped backtick runs",
          "[core][compositor][live-markdown]") {
    Flowgraph flowgraph;
    REQUIRE(flowgraph.environment().set("telemetry", Parser::Map{{"value", U64{42}}}) == Result::SUCCESS);

    const struct {
        const char* source;
        const char* expanded;
    } cases[] = {
        {"Code: `${env.telemetry.value}", "Code: `42"},
        {"Code: ``${env.telemetry.value}`", "Code: ``42`"},
        {"Code: ``${env.telemetry.value}```", "Code: ``42```"},
        {"Code: \\`${env.telemetry.value}", "Code: \\`42"},
        {"Code: `` unmatched `${env.telemetry.value}` ${env.telemetry.value}",
         "Code: `` unmatched `${env.telemetry.value}` 42"},
    };
    for (const auto& entry : cases) {
        CAPTURE(entry.source);
        const LiveMarkdown markdown(entry.source);
        CHECK(markdown.dynamic());
        CHECK(markdown.expand(flowgraph.environment()) == entry.expanded);
    }
}

TEST_CASE("Live markdown preserves invalid formats regardless of value availability",
          "[core][compositor][live-markdown]") {
    Flowgraph flowgraph;
    REQUIRE(flowgraph.environment().set("telemetry", Parser::Map{
        {"value", U64{42}},
        {"values", std::vector<U64>{1, 2}},
        {"empty", std::vector<U64>{}},
    }) == Result::SUCCESS);

    for (const std::string path : {
        "env.missing.value",
        "env.missing.values[*]",
        "env.telemetry.missing",
        "env.telemetry.absent[*]",
        "env.telemetry.empty[*]",
        "env.telemetry.values[*]",
        "env.telemetry.value",
    }) {
        for (const std::string spec : {"bogus", ">65", ".65f"}) {
            CAPTURE(path, spec);
            const std::string source = "${" + path + ":" + spec + "}";
            const LiveMarkdown literal(source);
            CHECK_FALSE(literal.dynamic());
            CHECK(literal.expand(flowgraph.environment()) == source);

            const LiveMarkdown mixed("Value: ${env.telemetry.value}; invalid: " + source);
            CHECK(mixed.dynamic());
            CHECK(mixed.expand(flowgraph.environment()) == "Value: 42; invalid: " + source);
        }
    }
}

TEST_CASE("Live markdown still compiles valid formats before looking up values",
          "[core][compositor][live-markdown]") {
    Flowgraph flowgraph;
    REQUIRE(flowgraph.environment().set("telemetry", Parser::Map{
        {"value", U64{42}},
        {"floating", F64{1.5}},
        {"values", std::vector<U64>{1, 2}},
        {"empty", std::vector<U64>{}},
    }) == Result::SUCCESS);

    const struct {
        const char* source;
        const char* expanded;
    } cases[] = {
        {"${env.telemetry.value:}", "42"},
        {"${env.telemetry.value:x}", "2a"},
        {"${env.telemetry.floating:.2f}", "1\\.50"},
        {"${env.missing.value:x}", "--"},
        {"${env.telemetry.missing:.2f}", "--"},
        {"Values: ${env.telemetry.values[*]:x}", "Values: 1\nValues: 2"},
        {"${env.missing.values[*]:x}", ""},
        {"${env.telemetry.empty[*]:x}", ""},
        {"${env.telemetry.value:.2f}", "${env.telemetry.value:.2f}"},
    };
    for (const auto& entry : cases) {
        CAPTURE(entry.source);
        const LiveMarkdown markdown(entry.source);
        CHECK(markdown.dynamic());
        CHECK(markdown.expand(flowgraph.environment()) == entry.expanded);
    }
}
