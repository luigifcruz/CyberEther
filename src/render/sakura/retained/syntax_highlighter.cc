#include "syntax_highlighter.hh"

#include <jetstream/logger.hh>

#include <algorithm>
#include <cctype>
#include <utility>

extern "C" const TSLanguage* tree_sitter_python(void);
extern "C" const TSLanguage* tree_sitter_markdown(void);
extern "C" const TSLanguage* tree_sitter_yaml(void);
extern "C" const TSLanguage* tree_sitter_bash(void);
extern "C" const TSLanguage* tree_sitter_cpp(void);
extern "C" const TSLanguage* tree_sitter_json(void);

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
  (attribute attribute: (identifier) @property)
  (call function: (attribute attribute: (identifier) @function.method))
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

constexpr std::string_view kYamlHighlightQuery = R"(
  [
    (double_quote_scalar) (single_quote_scalar) (block_scalar) (string_scalar)
  ] @string
  [ (integer_scalar) (float_scalar) ] @number
  (boolean_scalar) @boolean
  (null_scalar) @constant.builtin
  [ (anchor_name) (alias_name) ] @label
  (tag) @type
  [ (yaml_directive) (tag_directive) (reserved_directive) ] @attribute
  (block_mapping_pair
    key: (flow_node [ (double_quote_scalar) (single_quote_scalar) ] @property))
  (block_mapping_pair
    key: (flow_node (plain_scalar (string_scalar) @property)))
  (flow_mapping
    (_ key: (flow_node [ (double_quote_scalar) (single_quote_scalar) ] @property)))
  (flow_mapping
    (_ key: (flow_node (plain_scalar (string_scalar) @property))))
  [ "," "-" ":" ">" "?" "|" ] @punctuation.delimiter
  [ "[" "]" "{" "}" ] @punctuation.bracket
  [ "*" "&" "---" "..." ] @punctuation.special
  (comment) @comment
)";

constexpr std::string_view kBashHighlightQuery = R"(
  [ (string) (raw_string) (heredoc_body) (heredoc_start) ] @string
  (variable_name) @property
  [ (simple_expansion) (expansion) ] @property
  (command_name) @function
  (function_definition name: (word) @function)
  [ (number) (file_descriptor) ] @number
  [
    "case" "do" "done" "elif" "else" "esac" "export" "fi" "for" "function" "if" "in"
    "select" "then" "unset" "until" "while" "declare" "local" "readonly" "typeset"
  ] @keyword
  [ "$" "&&" "||" ">" ">>" "<" "|" ] @operator
  (comment) @comment
)";

constexpr std::string_view kCppHighlightQuery = R"(
  [ (type_identifier) (primitive_type) (sized_type_specifier) (auto) ] @type
  (field_identifier) @property
  (number_literal) @number
  [ (string_literal) (raw_string_literal) (char_literal) (system_lib_string) ] @string
  [ (true) (false) (null) (this) ] @constant.builtin
  [
    "break" "case" "catch" "class" "co_await" "co_return" "co_yield" "concept" "const"
    "consteval" "constexpr" "constinit" "continue" "default" "delete" "do" "else" "enum"
    "explicit" "extern" "final" "for" "friend" "if" "inline" "mutable" "namespace" "new"
    "noexcept" "override" "private" "protected" "public" "requires" "return" "sizeof"
    "static" "struct" "switch" "template" "throw" "try" "typedef" "typename" "union"
    "using" "virtual" "volatile" "while"
    "#define" "#elif" "#else" "#endif" "#if" "#ifdef" "#ifndef" "#include"
  ] @keyword
  (preproc_directive) @keyword
  [
    "--" "-" "-=" "->" "=" "!=" "*" "&" "&&" "+" "++" "+=" "<" "==" ">" "||" "!" "/" "%"
    "<=" ">=" "<<" ">>" "::"
  ] @operator
  (call_expression function: (identifier) @function)
  (call_expression function: (field_expression field: (field_identifier) @function))
  (call_expression function: (qualified_identifier name: (identifier) @function))
  (function_declarator declarator: (identifier) @function)
  (function_declarator declarator: (field_identifier) @function)
  (function_declarator declarator: (qualified_identifier name: (identifier) @function))
  (template_function name: (identifier) @function)
  (template_method name: (field_identifier) @function)
  (preproc_function_def name: (identifier) @function)
  (comment) @comment
)";

constexpr std::string_view kJsonHighlightQuery = R"(
  (string) @string
  (number) @number
  [ (null) (true) (false) ] @constant.builtin
  (pair key: (string) @property)
  (comment) @comment
)";

struct LanguageTag {
    std::string_view tag;
    Language language;
};

constexpr LanguageTag kLanguageTags[] = {
    {"python", Language::Python}, {"py", Language::Python}, {"python3", Language::Python},
    {"markdown", Language::Markdown}, {"md", Language::Markdown},
    {"yaml", Language::Yaml}, {"yml", Language::Yaml},
    {"bash", Language::Bash}, {"sh", Language::Bash}, {"shell", Language::Bash}, {"zsh", Language::Bash},
    {"cpp", Language::Cpp}, {"c++", Language::Cpp}, {"cc", Language::Cpp}, {"cxx", Language::Cpp},
    {"hpp", Language::Cpp}, {"hh", Language::Cpp}, {"hxx", Language::Cpp}, {"c", Language::Cpp},
    {"h", Language::Cpp},
    {"json", Language::Json}, {"jsonc", Language::Json},
};

struct SyntaxGrammar {
    std::string_view name;
    const TSLanguage* (*grammar)();
    std::string_view highlightQuery;
};

const SyntaxGrammar& GrammarFor(Language language) {
    static const SyntaxGrammar grammars[SyntaxHighlighter::LanguageCount] = {
        {"python", tree_sitter_python, kPythonHighlightQuery},
        {"markdown", tree_sitter_markdown, kMarkdownHighlightQuery},
        {"yaml", tree_sitter_yaml, kYamlHighlightQuery},
        {"bash", tree_sitter_bash, kBashHighlightQuery},
        {"cpp", tree_sitter_cpp, kCppHighlightQuery},
        {"json", tree_sitter_json, kJsonHighlightQuery},
    };
    return grammars[static_cast<U64>(language)];
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
    if (capture.starts_with("constant") || capture == "boolean" || capture == "text.uri") {
        return Style::Constant;
    }
    if (capture == "keyword" || capture == "attribute") {
        return Style::Keyword;
    }
    if (capture == "operator" || capture.starts_with("punctuation")) {
        return Style::Operator;
    }
    if (capture.starts_with("function") || capture == "text.title") {
        return Style::Function;
    }
    if (capture == "type" || capture == "label" || capture == "text.reference") {
        return Style::Type;
    }
    if (capture == "property") {
        return Style::Property;
    }
    return Style::Default;
}

struct PatternCapture {
    U32 pattern = 0;
    TSNode node;
    StyleId style = Style::Default;
};

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
    for (const auto& entry : kLanguageTags) {
        if (entry.tag == lower) {
            return entry.language;
        }
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
    for (TSQuery* query : queries) {
        if (query) {
            ts_query_delete(query);
        }
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
    applyQuery(queries[static_cast<U64>(language)], styles, lines, ts_tree_root_node(tree));
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
        activeLanguage = language;
        activeLanguageValid = true;
    }
    const U64 index = static_cast<U64>(language);
    if (queries[index]) {
        return true;
    }
    if (queryFailed[index]) {
        return false;
    }
    U32 errorOffset = 0;
    TSQueryError errorType = TSQueryErrorNone;
    queries[index] = ts_query_new(grammar.grammar(), grammar.highlightQuery.data(),
                                  static_cast<U32>(grammar.highlightQuery.size()), &errorOffset, &errorType);
    if (!queries[index]) {
        queryFailed[index] = true;
        JST_ERROR("[SAKURA] Failed to compile {} highlight query at byte {} (error {}).",
                  grammar.name, errorOffset, static_cast<U32>(errorType));
        return false;
    }
    return true;
}

void SyntaxHighlighter::applyQuery(const TSQuery* query, std::vector<std::vector<StyleId>>& styles,
                                   const std::vector<std::string>& lines, TSNode root) const {
    TSQueryCursor* cursor = ts_query_cursor_new();
    if (!cursor) {
        return;
    }
    ts_query_cursor_exec(cursor, query, root);
    std::vector<PatternCapture> captures;
    TSQueryMatch match;
    while (ts_query_cursor_next_match(cursor, &match)) {
        for (U16 i = 0; i < match.capture_count; ++i) {
            const TSQueryCapture& capture = match.captures[i];
            U32 nameLength = 0;
            const char* name = ts_query_capture_name_for_id(query, capture.index, &nameLength);
            if (name) {
                captures.push_back({match.pattern_index, capture.node, StyleForCapture({name, nameLength})});
            }
        }
    }
    ts_query_cursor_delete(cursor);
    std::stable_sort(captures.begin(), captures.end(), [](const PatternCapture& a, const PatternCapture& b) {
        return a.pattern < b.pattern;
    });
    for (const auto& capture : captures) {
        ApplyRange(styles, lines, capture.node, capture.style);
    }
}

}  // namespace Jetstream::Sakura::Retained
