#include <catch2/catch_test_macros.hpp>

#include "render/sakura/retained/syntax_highlighter.hh"

#include <string>
#include <vector>

using namespace Jetstream;
using Highlighter = Sakura::Retained::SyntaxHighlighter;

TEST_CASE("Fence tags resolve to highlighter languages",
          "[core][sakura][syntax]") {
    CHECK(Highlighter::LanguageForTag("python") == Highlighter::Language::Python);
    CHECK(Highlighter::LanguageForTag("Py") == Highlighter::Language::Python);
    CHECK(Highlighter::LanguageForTag("md") == Highlighter::Language::Markdown);
    CHECK_FALSE(Highlighter::LanguageForTag("rust").has_value());
    CHECK_FALSE(Highlighter::LanguageForTag("").has_value());
}

TEST_CASE("Python sources receive keyword, function, and number styles",
          "[core][sakura][syntax]") {
    Highlighter highlighter;
    const std::vector<std::string> lines = {"def f():", "    return 1  # done"};
    const auto styles = highlighter.highlight(lines, Highlighter::Language::Python);

    REQUIRE(styles.size() == 2);
    REQUIRE(styles[0].size() == lines[0].size());
    CHECK(styles[0][0] == Highlighter::Keyword);
    CHECK(styles[0][4] == Highlighter::Function);
    CHECK(styles[1][4] == Highlighter::Keyword);
    CHECK(styles[1][11] == Highlighter::Number);
    CHECK(styles[1][14] == Highlighter::Comment);
}

TEST_CASE("Fence tags cover yaml, shell, c family, and json aliases",
          "[core][sakura][syntax]") {
    CHECK(Highlighter::LanguageForTag("yml") == Highlighter::Language::Yaml);
    CHECK(Highlighter::LanguageForTag("sh") == Highlighter::Language::Bash);
    CHECK(Highlighter::LanguageForTag("zsh") == Highlighter::Language::Bash);
    CHECK(Highlighter::LanguageForTag("c") == Highlighter::Language::Cpp);
    CHECK(Highlighter::LanguageForTag("C++") == Highlighter::Language::Cpp);
    CHECK(Highlighter::LanguageForTag("jsonc") == Highlighter::Language::Json);
}

TEST_CASE("YAML sources style keys, scalars, punctuation, and comments",
          "[core][sakura][syntax]") {
    Highlighter highlighter;
    const std::vector<std::string> lines = {"graph:", "  - name: qpsk  # x", "    rate: 2.5"};
    const auto styles = highlighter.highlight(lines, Highlighter::Language::Yaml);

    REQUIRE(styles.size() == 3);
    CHECK(styles[0][0] == Highlighter::Property);
    CHECK(styles[0][5] == Highlighter::Operator);
    CHECK(styles[1][2] == Highlighter::Operator);
    CHECK(styles[1][4] == Highlighter::Property);
    CHECK(styles[1][10] == Highlighter::String);
    CHECK(styles[1][16] == Highlighter::Comment);
    CHECK(styles[2][10] == Highlighter::Number);
}

TEST_CASE("Bash sources style keywords, variables, commands, and comments",
          "[core][sakura][syntax]") {
    Highlighter highlighter;
    const std::vector<std::string> lines = {"export PATH=\"$HOME/bin\"  # add", "ls -la | grep x"};
    const auto styles = highlighter.highlight(lines, Highlighter::Language::Bash);

    REQUIRE(styles.size() == 2);
    CHECK(styles[0][0] == Highlighter::Keyword);
    CHECK(styles[0][7] == Highlighter::Property);
    CHECK(styles[0][12] == Highlighter::String);
    CHECK(styles[0][14] == Highlighter::Property);
    CHECK(styles[0][25] == Highlighter::Comment);
    CHECK(styles[1][0] == Highlighter::Function);
    CHECK(styles[1][7] == Highlighter::Operator);
}

TEST_CASE("C++ sources style directives, types, functions, and literals",
          "[core][sakura][syntax]") {
    Highlighter highlighter;
    const std::vector<std::string> lines = {"#include <vector>", "int main() { return 42; }  // c"};
    const auto styles = highlighter.highlight(lines, Highlighter::Language::Cpp);

    REQUIRE(styles.size() == 2);
    CHECK(styles[0][0] == Highlighter::Keyword);
    CHECK(styles[0][9] == Highlighter::String);
    CHECK(styles[1][0] == Highlighter::Type);
    CHECK(styles[1][4] == Highlighter::Function);
    CHECK(styles[1][13] == Highlighter::Keyword);
    CHECK(styles[1][20] == Highlighter::Number);
    CHECK(styles[1][27] == Highlighter::Comment);
}

TEST_CASE("JSON sources style keys apart from string values",
          "[core][sakura][syntax]") {
    Highlighter highlighter;
    const std::vector<std::string> lines = {"{\"rate\": 2.5, \"on\": true, \"name\": \"x\"}"};
    const auto styles = highlighter.highlight(lines, Highlighter::Language::Json);

    REQUIRE(styles.size() == 1);
    CHECK(styles[0][2] == Highlighter::Property);
    CHECK(styles[0][9] == Highlighter::Number);
    CHECK(styles[0][15] == Highlighter::Property);
    CHECK(styles[0][20] == Highlighter::Constant);
    CHECK(styles[0][35] == Highlighter::String);
}

TEST_CASE("Switching languages keeps each compiled query usable",
          "[core][sakura][syntax]") {
    Highlighter highlighter;
    const std::vector<std::string> python = {"def f():"};
    const std::vector<std::string> json = {"{\"a\": 1}"};
    for (int i = 0; i < 2; ++i) {
        CHECK(highlighter.highlight(python, Highlighter::Language::Python)[0][0] == Highlighter::Keyword);
        CHECK(highlighter.highlight(json, Highlighter::Language::Json)[0][6] == Highlighter::Number);
    }
}
