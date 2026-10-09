#include <jetstream/render/sakura/components/retained/text_editor.hh>

#include "../../retained/syntax_highlighter.hh"

#include <algorithm>
#include <cctype>
#include <optional>
#include <span>
#include <string_view>
#include <utility>
#include <vector>

namespace Jetstream::Sakura::Retained {

namespace {

using StyleId = TextGrid::StyleId;
using Position = TextGrid::Position;

constexpr U64 kTabSize = 4;

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

SyntaxHighlighter::Language SyntaxLanguage(TextEditor::Language language) {
    return language == TextEditor::Language::Markdown ? SyntaxHighlighter::Language::Markdown
                                                       : SyntaxHighlighter::Language::Python;
}

std::span<const std::string_view> BlockOpenerNodes(TextEditor::Language language) {
    if (language == TextEditor::Language::Python) {
        return kPythonBlockOpenerNodes;
    }
    return {};
}

std::span<const std::string_view> IndentContainerNodes(TextEditor::Language language) {
    if (language == TextEditor::Language::Python) {
        return kPythonIndentContainerNodes;
    }
    return {};
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

}  // namespace

struct TextEditor::Impl {
    Config config;
    TextGrid grid;
    SyntaxHighlighter highlighter;
    U64 lastRevision = 0;
    U64 styleRevision = 0;

    bool styleIsCommentOrString(StyleId id) const {
        return SyntaxHighlighter::IsCommentOrString(id);
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
        if (!highlighter.rootNode(lines, lastRevision, SyntaxLanguage(config.language), root)) {
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
        const auto openers = BlockOpenerNodes(config.language);
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
        const auto containers = IndentContainerNodes(config.language);
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
        const auto& styles = highlighter.styles(lines, lastRevision, SyntaxLanguage(config.language));
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
        const auto& styles = highlighter.styles(lines, lastRevision, SyntaxLanguage(config.language));
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
        if (!highlighter.rootNode(lines, lastRevision, SyntaxLanguage(config.language), root)) {
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
    if (impl->config.language != config.language) {
        ++impl->styleRevision;
    }
    impl->config = std::move(config);

    impl->grid.update({
        .id = impl->config.id + ":grid",
        .value = impl->config.value,
        .editable = true,
        .fontSize = impl->config.fontSize,
        .lineHeight = impl->config.lineHeight,
        .fontName = impl->config.fontName,
        .monospace = impl->config.monospace,
        .lineNumbers = impl->config.lineNumbers,
        .showActiveLine = impl->config.showActiveLine,
        .wrap = impl->config.wrap,
        .padding = impl->config.padding,
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
        .styleScales = impl->config.styleScales,
        .styleRevision = impl->styleRevision,
        .styler = [impl = this->impl.get()](const std::vector<std::string>& lines, U64 revision)
                      -> const std::vector<std::vector<StyleId>>& {
            impl->lastRevision = revision;
            return impl->highlighter.styles(lines, revision, SyntaxLanguage(impl->config.language));
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

const TextGrid::Metrics& TextEditor::metrics() const {
    return impl->grid.metrics();
}

Extent2D<F32> TextEditor::measure(const Context& ctx, Extent2D<F32> available) {
    return measureChild(this->impl->grid, ctx, available);
}

void TextEditor::layout(const Context& ctx) {
    layoutChild(ctx, this->impl->grid, frame());
}

}  // namespace Jetstream::Sakura::Retained
