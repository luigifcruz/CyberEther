#include "syntax_highlighter.hh"

#include <jetstream/logger.hh>

#include <algorithm>
#include <cctype>
#include <utility>

extern "C" const TSLanguage* tree_sitter_python(void);
extern "C" const TSLanguage* tree_sitter_markdown(void);

namespace Jetstream::Sakura::Retained {

namespace {

using StyleId = SyntaxHighlighter::StyleId;
using Style = SyntaxHighlighter::Style;
using Language = SyntaxHighlighter::Language;

constexpr std::string_view kPythonHighlightQuery = R"(
  (comment) @comment
  (string) @string
  (escape_sequence) @escape
  [ (integer) (float) ] @number
  [ (none) (true) (false) ] @constant.builtin
  (function_definition name: (identifier) @function)
  (class_definition name: (identifier) @type)
  (call function: (identifier) @function)
  (call function: (attribute attribute: (identifier) @function.method))
  (attribute attribute: (identifier) @property)
  (type (identifier) @type)
  [
    "as" "assert" "async" "await" "break" "class" "continue" "def" "del"
    "elif" "else" "except" "finally" "for" "from" "global" "if" "import"
    "lambda" "nonlocal" "pass" "raise" "return" "try" "while" "with" "yield"
    "match" "case"
  ] @keyword
  [
    "and" "in" "is" "not" "or" "is not" "not in" "=" ":=" "==" "!=" "<" "<=" ">" ">="
    "+" "-" "*" "**" "/" "//" "%" "&" "|" "^" "~" "<<" ">>" "+=" "-=" "*="
    "**=" "/=" "//=" "%=" "&=" "|=" "^=" "<<=" ">>=" "->"
  ] @operator
)";

constexpr std::string_view kMarkdownHighlightQuery = R"(
  (atx_heading (inline) @text.title)
  (setext_heading (paragraph) @text.title)
  [
    (atx_h1_marker) (atx_h2_marker) (atx_h3_marker) (atx_h4_marker)
    (atx_h5_marker) (atx_h6_marker) (setext_h1_underline) (setext_h2_underline)
  ] @punctuation.special
  [ (link_title) (indented_code_block) (fenced_code_block) ] @text.literal
  (fenced_code_block_delimiter) @punctuation.delimiter
  (code_fence_content) @none
  (link_destination) @text.uri
  (link_label) @text.reference
  [
    (list_marker_plus) (list_marker_minus) (list_marker_star) (list_marker_dot)
    (list_marker_parenthesis) (thematic_break)
  ] @punctuation.special
  [ (block_continuation) (block_quote_marker) ] @punctuation.special
  (backslash_escape) @string.escape
)";

struct SyntaxGrammar {
    std::string_view name;
    const TSLanguage* (*grammar)();
    std::string_view highlightQuery;
};

const SyntaxGrammar& GrammarFor(Language language) {
    static const SyntaxGrammar python = {"python", tree_sitter_python, kPythonHighlightQuery};
    static const SyntaxGrammar markdown = {"markdown", tree_sitter_markdown, kMarkdownHighlightQuery};
    return language == Language::Markdown ? markdown : python;
}

std::string JoinLines(const std::vector<std::string>& lines) {
    std::string value;
    for (U64 i = 0; i < lines.size(); ++i) {
        if (i > 0) {
            value += '\n';
        }
        value += lines[i];
    }
    return value;
}

StyleId StyleForCapture(std::string_view capture) {
    if (capture == "comment") {
        return Style::Comment;
    }
    if (capture == "string" || capture == "escape" ||
        capture == "text.literal" || capture == "string.escape") {
        return Style::String;
    }
    if (capture == "number") {
        return Style::Number;
    }
    if (capture.starts_with("constant") || capture == "text.uri") {
        return Style::Constant;
    }
    if (capture == "keyword") {
        return Style::Keyword;
    }
    if (capture == "operator" || capture.starts_with("punctuation")) {
        return Style::Operator;
    }
    if (capture.starts_with("function") || capture == "text.title") {
        return Style::Function;
    }
    if (capture == "type" || capture == "text.reference") {
        return Style::Type;
    }
    if (capture == "property") {
        return Style::Property;
    }
    return Style::Default;
}

std::vector<std::vector<StyleId>> BlankStyles(const std::vector<std::string>& lines) {
    std::vector<std::vector<StyleId>> styles;
    styles.reserve(lines.size());
    for (const auto& line : lines) {
        styles.emplace_back(line.size(), Style::Default);
    }
    return styles;
}

void ApplyRange(std::vector<std::vector<StyleId>>& styles, const std::vector<std::string>& lines,
                TSNode node, StyleId style) {
    const TSPoint start = ts_node_start_point(node);
    const TSPoint end = ts_node_end_point(node);
    for (U64 row = start.row; row <= end.row && row < styles.size(); ++row) {
        const U64 lineSize = std::min<U64>(styles[row].size(), lines[row].size());
        const U64 startColumn = row == start.row ? std::min<U64>(start.column, lineSize) : 0;
        const U64 endColumn = row == end.row ? std::min<U64>(end.column, lineSize) : lineSize;
        for (U64 column = startColumn; column < endColumn; ++column) {
            styles[row][column] = style;
        }
    }
}

}  // namespace

std::optional<Language> SyntaxHighlighter::LanguageForTag(std::string_view tag) {
    std::string lower(tag);
    std::transform(lower.begin(), lower.end(), lower.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    if (lower == "python" || lower == "py" || lower == "python3") {
        return Language::Python;
    }
    if (lower == "markdown" || lower == "md") {
        return Language::Markdown;
    }
    return std::nullopt;
}

bool SyntaxHighlighter::IsCommentOrString(StyleId id) {
    return id == Style::Comment || id == Style::String;
}

SyntaxHighlighter::SyntaxHighlighter() = default;

SyntaxHighlighter::~SyntaxHighlighter() {
    if (tree) {
        ts_tree_delete(tree);
    }
    if (query) {
        ts_query_delete(query);
    }
    if (parser) {
        ts_parser_delete(parser);
    }
}

const std::vector<std::vector<StyleId>>& SyntaxHighlighter::styles(const std::vector<std::string>& lines,
                                                                   U64 revision, Language language) {
    if (cacheValid && revision == cachedRevision && language == cachedLanguage) {
        return cachedStyles;
    }
    cachedStyles = highlight(lines, language);
    cachedRevision = revision;
    cachedLanguage = language;
    cacheValid = true;
    return cachedStyles;
}

std::vector<std::vector<StyleId>> SyntaxHighlighter::highlight(const std::vector<std::string>& lines,
                                                               Language language) {
    cacheValid = false;
    auto styles = BlankStyles(lines);
    if (!ensureTreeSitter(language)) {
        return styles;
    }
    const std::string source = JoinLines(lines);
    TSTree* nextTree = ts_parser_parse_string(parser, nullptr, source.c_str(), static_cast<U32>(source.size()));
    if (!nextTree) {
        return styles;
    }
    if (tree) {
        ts_tree_delete(tree);
    }
    tree = nextTree;
    applyQuery(styles, lines, ts_tree_root_node(tree));
    return styles;
}

bool SyntaxHighlighter::rootNode(const std::vector<std::string>& lines, U64 revision, Language language,
                                 TSNode& root) {
    (void)styles(lines, revision, language);
    if (!tree) {
        return false;
    }
    root = ts_tree_root_node(tree);
    return !ts_node_is_null(root);
}

bool SyntaxHighlighter::ensureTreeSitter(Language language) {
    const auto& grammar = GrammarFor(language);
    if (!parser) {
        parser = ts_parser_new();
    }
    if (!parser) {
        return false;
    }
    if (!activeLanguageValid || activeLanguage != language) {
        if (!ts_parser_set_language(parser, grammar.grammar())) {
            return false;
        }
        if (tree) {
            ts_tree_delete(tree);
            tree = nullptr;
        }
        if (query) {
            ts_query_delete(query);
            query = nullptr;
        }
        activeLanguage = language;
        activeLanguageValid = true;
    }
    if (query) {
        return true;
    }
    U32 errorOffset = 0;
    TSQueryError errorType = TSQueryErrorNone;
    query = ts_query_new(grammar.grammar(), grammar.highlightQuery.data(),
                         static_cast<U32>(grammar.highlightQuery.size()), &errorOffset, &errorType);
    if (!query) {
        JST_ERROR("[SAKURA] Failed to compile {} highlight query at byte {} (error {}).",
                  grammar.name, errorOffset, static_cast<U32>(errorType));
        return false;
    }
    return true;
}

void SyntaxHighlighter::applyQuery(std::vector<std::vector<StyleId>>& styles, const std::vector<std::string>& lines,
                                   TSNode root) const {
    TSQueryCursor* cursor = ts_query_cursor_new();
    if (!cursor) {
        return;
    }
    ts_query_cursor_exec(cursor, query, root);
    TSQueryMatch match;
    while (ts_query_cursor_next_match(cursor, &match)) {
        for (U16 i = 0; i < match.capture_count; ++i) {
            const TSQueryCapture& capture = match.captures[i];
            U32 nameLength = 0;
            const char* name = ts_query_capture_name_for_id(query, capture.index, &nameLength);
            if (name) {
                ApplyRange(styles, lines, capture.node, StyleForCapture({name, nameLength}));
            }
        }
    }
    ts_query_cursor_delete(cursor);
}

}  // namespace Jetstream::Sakura::Retained
