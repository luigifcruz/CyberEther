#include <jetstream/render/sakura/components/retained/text_editor.hh>

#include <jetstream/logger.hh>

#include <tree_sitter/api.h>

#include <algorithm>
#include <cctype>
#include <optional>
#include <span>
#include <string_view>
#include <utility>
#include <vector>

extern "C" const TSLanguage* tree_sitter_python(void);
extern "C" const TSLanguage* tree_sitter_markdown(void);

namespace Jetstream::Sakura::Retained {

namespace {

using StyleId = TextGrid::StyleId;
using Position = TextGrid::Position;

constexpr U64 kTabSize = 4;

enum Style : StyleId {
    StyleDefault = 0,
    StyleComment = 1,
    StyleKeyword = 2,
    StyleString = 3,
    StyleNumber = 4,
    StyleFunction = 5,
    StyleType = 6,
    StyleConstant = 7,
    StyleOperator = 8,
    StyleProperty = 9,
};

constexpr std::string_view kPythonBlockOpenerNodes[] = {
    "if_statement",
    "elif_clause",
    "else_clause",
    "for_statement",
    "while_statement",
    "try_statement",
    "except_clause",
    "finally_clause",
    "with_statement",
    "function_definition",
    "class_definition",
    "match_statement",
    "case_clause",
};

constexpr std::string_view kPythonIndentContainerNodes[] = {
    "argument_list",
    "parameters",
    "list",
    "dictionary",
    "set",
    "tuple",
    "parenthesized_expression",
    "subscript",
    "list_comprehension",
    "dictionary_comprehension",
    "set_comprehension",
    "generator_expression",
};

constexpr std::string_view kPythonTerminalStatementKeywords[] = {
    "pass",
    "break",
    "continue",
    "return",
};

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
    std::span<const std::string_view> blockOpenerNodes;
    std::span<const std::string_view> indentContainerNodes;
};

const SyntaxGrammar& GrammarFor(TextEditor::Language language) {
    static const SyntaxGrammar python = {"python", tree_sitter_python, kPythonHighlightQuery,
                                         kPythonBlockOpenerNodes, kPythonIndentContainerNodes};
    static const SyntaxGrammar markdown = {"markdown", tree_sitter_markdown, kMarkdownHighlightQuery, {}, {}};
    return language == TextEditor::Language::Markdown ? markdown : python;
}

bool ContainsNodeType(std::span<const std::string_view> values, std::string_view value) {
    for (const auto& candidate : values) {
        if (candidate == value) {
            return true;
        }
    }
    return false;
}

bool IsSelectableSyntaxNode(std::string_view type) {
    return type != "module" && type != "ERROR" && type != "comment";
}

bool IsWordCharacter(char character) {
    return std::isalnum(static_cast<unsigned char>(character)) || character == '_';
}

bool StartsWithKeyword(std::span<const std::string_view> keywords, std::string_view content) {
    for (const auto& keyword : keywords) {
        if (content.size() >= keyword.size() && content.substr(0, keyword.size()) == keyword &&
            (content.size() == keyword.size() || !IsWordCharacter(content[keyword.size()]))) {
            return true;
        }
    }
    return false;
}

U64 LeadingWhitespaceColumn(const std::string& line) {
    U64 column = 0;
    while (column < line.size() && (line[column] == ' ' || line[column] == '\t')) {
        ++column;
    }
    return column;
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
        return StyleComment;
    }
    if (capture == "string" || capture == "escape" ||
        capture == "text.literal" || capture == "string.escape") {
        return StyleString;
    }
    if (capture == "number") {
        return StyleNumber;
    }
    if (capture.starts_with("constant") || capture == "text.uri") {
        return StyleConstant;
    }
    if (capture == "keyword") {
        return StyleKeyword;
    }
    if (capture == "operator" || capture.starts_with("punctuation")) {
        return StyleOperator;
    }
    if (capture.starts_with("function") || capture == "text.title") {
        return StyleFunction;
    }
    if (capture == "type" || capture == "text.reference") {
        return StyleType;
    }
    if (capture == "property") {
        return StyleProperty;
    }
    return StyleDefault;
}

struct SyntaxHighlighter {
    TSParser* parser = nullptr;
    TSQuery* query = nullptr;
    TSTree* tree = nullptr;
    std::vector<std::vector<StyleId>> cachedStyles;
    U64 cachedRevision = 0;
    TextEditor::Language cachedLanguage = TextEditor::Language::Python;
    TextEditor::Language activeLanguage = TextEditor::Language::Python;
    bool cacheValid = false;
    bool activeLanguageValid = false;

    ~SyntaxHighlighter() {
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

    const std::vector<std::vector<StyleId>>& styles(const std::vector<std::string>& lines,
                                                    U64 revision,
                                                    TextEditor::Language language) {
        if (cacheValid && revision == cachedRevision && language == cachedLanguage) {
            return cachedStyles;
        }
        auto next = blankStyles(lines);
        applyTreeSitter(next, lines, language);
        cachedStyles = std::move(next);
        cachedRevision = revision;
        cachedLanguage = language;
        cacheValid = true;
        return cachedStyles;
    }

    bool rootNode(const std::vector<std::string>& lines, U64 revision,
                  TextEditor::Language language, TSNode& root) {
        (void)styles(lines, revision, language);
        if (!tree) {
            return false;
        }
        root = ts_tree_root_node(tree);
        return !ts_node_is_null(root);
    }

 private:
    static std::vector<std::vector<StyleId>> blankStyles(const std::vector<std::string>& lines) {
        std::vector<std::vector<StyleId>> styles;
        styles.reserve(lines.size());
        for (const auto& line : lines) {
            styles.emplace_back(line.size(), StyleDefault);
        }
        return styles;
    }

    void applyTreeSitter(std::vector<std::vector<StyleId>>& styles,
                         const std::vector<std::string>& lines,
                         TextEditor::Language language) {
        const std::string source = JoinLines(lines);
        if (!ensureTreeSitter(language)) {
            return;
        }
        TSTree* nextTree = ts_parser_parse_string(parser, nullptr, source.c_str(), static_cast<U32>(source.size()));
        if (!nextTree) {
            return;
        }
        if (tree) {
            ts_tree_delete(tree);
        }
        tree = nextTree;
        applyQuery(styles, lines, ts_tree_root_node(tree));
    }

    bool ensureTreeSitter(TextEditor::Language language) {
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

    static void applyRange(std::vector<std::vector<StyleId>>& styles,
                           const std::vector<std::string>& lines, TSNode node, StyleId style) {
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

    void applyQuery(std::vector<std::vector<StyleId>>& styles,
                    const std::vector<std::string>& lines, TSNode root) const {
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
                    applyRange(styles, lines, capture.node, StyleForCapture({name, nameLength}));
                }
            }
        }
        ts_query_cursor_delete(cursor);
    }
};

}  // namespace

struct TextEditor::Impl {
    Config config;
    TextGrid grid;
    SyntaxHighlighter highlighter;
    U64 lastRevision = 0;

    bool styleIsCommentOrString(StyleId id) const {
        return id == StyleComment || id == StyleString;
    }

    TSPoint pointFor(Position p) const {
        return {static_cast<U32>(p.line), static_cast<U32>(p.column)};
    }

    Position positionForPoint(const std::vector<std::string>& lines, TSPoint point) const {
        Position p{point.row, point.column};
        p.line = std::min<U64>(p.line, lines.empty() ? 0 : lines.size() - 1);
        if (p.line < lines.size()) {
            p.column = std::min<U64>(p.column, lines[p.line].size());
        }
        return p;
    }

    bool nodeAtPosition(const std::vector<std::string>& lines, Position position, TSNode& node, bool namedOnly) {
        TSNode root;
        if (!highlighter.rootNode(lines, lastRevision, config.language, root)) {
            return false;
        }
        const TSPoint point = pointFor(position);
        node = namedOnly ? ts_node_named_descendant_for_point_range(root, point, point)
                         : ts_node_descendant_for_point_range(root, point, point);
        return !ts_node_is_null(node);
    }

    bool hasBlockOpenerAncestor(const std::vector<std::string>& lines, Position position) {
        TSNode node;
        if (!nodeAtPosition(lines, position, node, false)) {
            return false;
        }
        const auto& openers = GrammarFor(config.language).blockOpenerNodes;
        while (!ts_node_is_null(node)) {
            if (ContainsNodeType(openers, ts_node_type(node))) {
                return true;
            }
            node = ts_node_parent(node);
        }
        return false;
    }

    bool isInsideIndentContainer(const std::vector<std::string>& lines, Position position) {
        TSNode node;
        if (!nodeAtPosition(lines, position, node, true)) {
            return false;
        }
        const auto& containers = GrammarFor(config.language).indentContainerNodes;
        const auto less = [](Position a, Position b) {
            return a.line != b.line ? a.line < b.line : a.column < b.column;
        };
        while (!ts_node_is_null(node)) {
            const Position start = positionForPoint(lines, ts_node_start_point(node));
            const Position end = positionForPoint(lines, ts_node_end_point(node));
            if (ContainsNodeType(containers, ts_node_type(node)) && less(start, position) && less(position, end)) {
                return true;
            }
            node = ts_node_parent(node);
        }
        return false;
    }

    bool hasUnclosedBracket(const std::vector<std::string>& lines, Position position) {
        const auto& styles = highlighter.styles(lines, lastRevision, config.language);
        I64 depth = 0;
        for (U64 lineIndex = 0; lineIndex <= position.line && lineIndex < lines.size(); ++lineIndex) {
            const auto& line = lines[lineIndex];
            const U64 endColumn = lineIndex == position.line ? std::min<U64>(position.column, line.size()) : line.size();
            for (U64 column = 0; column < endColumn; ++column) {
                if (lineIndex < styles.size() && column < styles[lineIndex].size() &&
                    styleIsCommentOrString(styles[lineIndex][column])) {
                    continue;
                }
                const char c = line[column];
                if (c == '(' || c == '[' || c == '{') {
                    ++depth;
                } else if ((c == ')' || c == ']' || c == '}') && depth > 0) {
                    --depth;
                }
            }
        }
        return depth > 0;
    }

    std::optional<std::string> computeNewlineIndent(const std::vector<std::string>& lines, Position cursor) {
        if (cursor.line >= lines.size()) {
            return std::nullopt;
        }
        const auto& line = lines[cursor.line];
        const std::string beforeCursor = line.substr(0, std::min<U64>(cursor.column, line.size()));
        std::string indent = line.substr(0, LeadingWhitespaceColumn(line));

        if (config.language == Language::Python) {
            const U64 lead = LeadingWhitespaceColumn(beforeCursor);
            if (StartsWithKeyword(kPythonTerminalStatementKeywords, std::string_view(beforeCursor).substr(lead))) {
                const U64 size = indent.size();
                indent.resize(size > 0 ? ((size - 1) / kTabSize) * kTabSize : 0, ' ');
                return indent;
            }
        }

        const auto lastContent = beforeCursor.find_last_not_of(" \t");
        if (lastContent == std::string::npos) {
            return indent;
        }
        const auto& styles = highlighter.styles(lines, lastRevision, config.language);
        const Position contentPos = {cursor.line, static_cast<U64>(lastContent)};
        const bool inCommentOrString = cursor.line < styles.size() && lastContent < styles[cursor.line].size() &&
                                       styleIsCommentOrString(styles[cursor.line][lastContent]);
        if (!inCommentOrString) {
            if (beforeCursor[lastContent] == ':') {
                if (hasBlockOpenerAncestor(lines, contentPos)) {
                    indent += std::string(kTabSize, ' ');
                }
            } else if (isInsideIndentContainer(lines, cursor) || hasUnclosedBracket(lines, cursor)) {
                indent += std::string(kTabSize, ' ');
            }
        }
        return indent;
    }

    std::optional<std::pair<Position, Position>> expandSelection(const std::vector<std::string>& lines,
                                                                 Position anchor, Position cursor) {
        const auto less = [](Position a, Position b) {
            return a.line != b.line ? a.line < b.line : a.column < b.column;
        };
        std::pair<Position, Position> basis = less(anchor, cursor) ? std::pair{anchor, cursor}
                                                                  : std::pair{cursor, anchor};
        const bool empty = basis.first == basis.second;

        TSNode root;
        if (!highlighter.rootNode(lines, lastRevision, config.language, root)) {
            return std::nullopt;
        }
        Position endPos = basis.second;
        if (!empty) {
            if (endPos.column > 0) {
                endPos.column -= 1;
            } else if (endPos.line > 0) {
                endPos.line -= 1;
                endPos.column = lines[endPos.line].size();
            }
        }
        TSNode node = ts_node_named_descendant_for_point_range(root, pointFor(basis.first), pointFor(endPos));
        while (!ts_node_is_null(node)) {
            const Position start = positionForPoint(lines, ts_node_start_point(node));
            const Position end = positionForPoint(lines, ts_node_end_point(node));
            const bool strictly = !less(basis.first, start) && !less(end, basis.second) &&
                                  (!(start == basis.first) || !(end == basis.second));
            if (strictly && IsSelectableSyntaxNode(ts_node_type(node))) {
                return std::pair{start, end};
            }
            node = ts_node_parent(node);
        }
        return std::nullopt;
    }
};

TextEditor::TextEditor() {
    this->impl = std::make_unique<Impl>();
    setClipsChildren(true);
    add(this->impl->grid);
}

TextEditor::~TextEditor() = default;

bool TextEditor::update(Config config) {
    impl->config = std::move(config);

    impl->grid.update({
        .id = impl->config.id + ":grid",
        .value = impl->config.value,
        .editable = true,
        .fontSize = impl->config.fontSize,
        .fontName = impl->config.fontName,
        .monospace = impl->config.monospace,
        .lineNumbers = impl->config.lineNumbers,
        .showActiveLine = true,
        .wrap = impl->config.wrap,
        .backgroundColorKey = impl->config.backgroundColorKey,
        .textColorKey = impl->config.textColorKey,
        .lineNumberColorKey = impl->config.lineNumberColorKey,
        .gutterSeparatorColorKey = impl->config.gutterSeparatorColorKey,
        .selectionColorKey = impl->config.selectionColorKey,
        .selectionMatchColorKey = impl->config.selectionMatchColorKey,
        .activeLineColorKey = impl->config.activeLineColorKey,
        .cursorColorKey = impl->config.cursorColorKey,
        .scrollbarTrackColorKey = impl->config.scrollbarTrackColorKey,
        .scrollbarThumbColorKey = impl->config.scrollbarThumbColorKey,
        .styleColorKeys = impl->config.styleColorKeys,
        .styleFonts = impl->config.styleFonts,
        .styleBackgroundColorKeys = impl->config.styleBackgroundColorKeys,
        .styler = [impl = this->impl.get()](const std::vector<std::string>& lines, U64 revision)
                      -> const std::vector<std::vector<StyleId>>& {
            impl->lastRevision = revision;
            return impl->highlighter.styles(lines, revision, impl->config.language);
        },
        .isStyleCommentOrString = [impl = this->impl.get()](StyleId id) {
            return impl->styleIsCommentOrString(id);
        },
        .onChange = impl->config.onChange,
        .onSubmit = impl->config.onSubmit,
        .computeNewlineIndent = [impl = this->impl.get()](const std::vector<std::string>& lines, Position cursor) {
            return impl->computeNewlineIndent(lines, cursor);
        },
        .expandSelection = [impl = this->impl.get()](const std::vector<std::string>& lines, Position anchor, Position cursor) {
            return impl->expandSelection(lines, anchor, cursor);
        },
    });
    return true;
}

Extent2D<F32> TextEditor::measure(const Context& ctx, Extent2D<F32> available) {
    return measureChild(this->impl->grid, ctx, available);
}

void TextEditor::layout(const Context& ctx) {
    layoutChild(ctx, this->impl->grid, frame());
}

}  // namespace Jetstream::Sakura::Retained
