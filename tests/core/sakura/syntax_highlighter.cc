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
