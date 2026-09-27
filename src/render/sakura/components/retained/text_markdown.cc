#include <jetstream/render/sakura/components/retained/text_markdown.hh>

#include <jetstream/platform.hh>
#include <jetstream/render/sakura/components/retained/box.hh>
#include <jetstream/render/sakura/components/retained/label.hh>
#include <jetstream/render/sakura/components/retained/text_grid.hh>

#include "../../context.hh"
#include "../../retained/helpers.hh"
#include "jetstream/render/tools/imgui_icons_ext.hh"
#include "../../retained/text_lines.hh"
#include "../../retained/syntax_highlighter.hh"
#include "../../retained/text_metrics.hh"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace Jetstream::Sakura::Retained {

namespace {

using StyleId = TextGrid::StyleId;
using Style = TextMarkdown::Style;

constexpr F32 kLineHeightRatio = Typography::BodyLineHeight;
constexpr F32 kParagraphGapRatio = 0.6f;
constexpr F32 kHeadingGapRatio = 1.0f;
constexpr F32 kAfterHeadingGapRatio = 0.2f;
constexpr F32 kListGapRatio = 0.15f;
constexpr F32 kIndentEmRatio = 1.4f;
constexpr F32 kCodePadRatio = 0.45f;
constexpr F32 kCodeCapHeightRatio = 0.73f;
constexpr U8 kCodeMinLineDigits = 2;
constexpr F32 kCodeGutterGapChars = 1.5f;
constexpr F32 kCodeBorderRatio = 1.0f / 15.0f;
constexpr F32 kRuleThicknessRatio = 0.12f;
constexpr F32 kRuleRowScale = 0.5f;
constexpr F32 kQuoteBarRatio = 0.18f;
constexpr F32 kQuoteIndentEmRatio = 0.4f;
constexpr F32 kBulletDotRatio = 0.22f;
constexpr F32 kBulletGapEmRatio = 0.75f;
constexpr F32 kMarkerGapEmRatio = 0.45f;
constexpr F32 kMarkerWidthEmRatio = 2.0f;
constexpr F32 kDecorationCornerRatio = 0.4f;
constexpr F32 kTableCellPadXRatio = 0.6f;
constexpr F32 kTableRowPadRatio = 0.4f;
constexpr F32 kTableBorderRatio = 0.07f;
constexpr F32 kTableMinColumnEm = 2.0f;
constexpr F32 kTableWrapSlack = 0.5f;
constexpr F32 kWidthTolerance = 0.5f;
constexpr F32 kTableNarrowPadShare = 0.25f;
constexpr F32 kFallbackGlyphWidthRatio = 0.5f;
constexpr U64 kTableShrinkIterations = 24;
constexpr F32 kCalloutPadEmRatio = 0.6f;
constexpr F32 kCalloutRadiusRatio = 0.8f;
constexpr F32 kCalloutFillAlpha = 0.10f;
constexpr F32 kCalloutBorderAlpha = 0.35f;
constexpr F32 kCalloutBorderRatio = 1.5f / 15.0f;
constexpr F32 kCalloutTitleScale = 1.0f;
constexpr F32 kCalloutTitleGapEmRatio = 0.25f;
constexpr U64 kMaxCallouts = 32;
constexpr U64 kCalloutTones = Style::CalloutTones;
constexpr std::array<std::string_view, kCalloutTones> kCalloutTags = {
    "[!NOTE]", "[!TIP]", "[!IMPORTANT]", "[!WARNING]", "[!CAUTION]", "[!ERROR]",
};
constexpr std::array<const char*, kCalloutTones> kCalloutTitles = {
    "Note", "Tip", "Important", "Warning", "Caution", "Error",
};
constexpr std::array<const char*, kCalloutTones> kCalloutIcons = {
    ICON_FA_CIRCLE_INFO, ICON_FA_LIGHTBULB, ICON_FA_COMMENT_DOTS,
    ICON_FA_TRIANGLE_EXCLAMATION, ICON_FA_HAND, ICON_FA_CIRCLE_XMARK,
};
constexpr std::array<const char*, kCalloutTones> kCalloutColorKeys = {
    "callout_note", "callout_tip", "callout_important", "callout_warning", "callout_caution", "callout_error",
};
constexpr U64 kMaxDecorations = 256;
constexpr U64 kMaxMarkerCharacters = 8;
constexpr U64 kGridLineSegments = 32;
constexpr const char* kBodyFont = TextMarkdown::BodyFont;
constexpr const char* kCodeFont = Typography::MonoFont;

constexpr std::array<std::pair<std::string_view, StyleId>, 4> kInlineDelimiters = {{
    {"`", Style::Code},
    {"***", Style::BoldItalic},
    {"**", Style::Bold},
    {"__", Style::Bold},
}};

struct TableSource {
    std::vector<I8> aligns;
    std::vector<std::vector<std::string>> rows;
};

struct Block {
    enum class Kind {
        Paragraph,
        Heading,
        Quote,
        Code,
        Rule,
        ListItem,
        Table,
        Callout,
    };

    Kind kind = Kind::Paragraph;
    std::string text;
    F32 fontSize = Typography::FontSize;
    F32 topGap = 0.0f;
    StyleId baseStyle = Style::Plain;
    F32 indent = 0.0f;
    std::string marker;
    TableSource table;
    StyleId tone = 0;
    std::optional<SyntaxHighlighter::Language> syntax;

    bool inlineParse() const { return kind != Kind::Code; }
};

struct LinkSpan {
    U64 start = 0;
    U64 end = 0;
    std::string url;
};

struct Deco {
    enum class Kind {
        Code,
        Quote,
        Rule,
        ListItem,
        Table,
        Callout,
    };

    Kind kind = Kind::Code;
    U64 first = 0;
    U64 last = 0;
    std::string marker;
    F32 indent = 0.0f;
    U64 table = 0;
    StyleId tone = 0;
    U8 digits = 0;
};

struct TableCell {
    U64 line = 0;
    std::string text;
    std::vector<StyleId> styles;
    F32 width = 0.0f;
};

struct Table {
    U64 columns = 0;
    U64 rows = 0;
    std::vector<I8> aligns;
    std::vector<TableCell> cells;
    std::vector<F32> columnX;
    F32 width = 0.0f;

    const TableCell& cell(U64 row, U64 column) const { return cells[row * columns + column]; }
};

bool IsFence(const std::string& line) {
    return line.rfind("```", 0) == 0;
}

std::string_view FenceTag(std::string_view line) {
    const U64 start = line.find_first_not_of("` \t");
    if (start == std::string_view::npos) {
        return {};
    }
    const U64 end = line.find_first_of(" \t{", start);
    return line.substr(start, end == std::string_view::npos ? std::string_view::npos : end - start);
}

bool IsRule(const std::string& line) {
    char mark = 0;
    U64 count = 0;
    for (const char c : line) {
        if (c == ' ' || c == '\t') {
            continue;
        }
        if (c != '-' && c != '*' && c != '_') {
            return false;
        }
        if (mark == 0) {
            mark = c;
        } else if (c != mark) {
            return false;
        }
        ++count;
    }
    return count >= 3;
}

U64 LeadingSpaces(const std::string& line) {
    U64 i = 0;
    while (i < line.size() && line[i] == ' ') {
        ++i;
    }
    return i;
}

U64 ListMarker(const std::string& line, std::string& marker) {
    marker.clear();
    const U64 i = LeadingSpaces(line);
    if (i + 1 < line.size() && (line[i] == '-' || line[i] == '*' || line[i] == '+') && line[i + 1] == ' ') {
        return i + 2;
    }
    U64 d = i;
    while (d < line.size() && line[d] >= '0' && line[d] <= '9') {
        ++d;
    }
    if (d > i && d + 1 < line.size() && line[d] == '.' && line[d + 1] == ' ') {
        marker = line.substr(i, d - i) + ".";
        return d + 2;
    }
    return 0;
}

bool IsListItem(const std::string& line) {
    std::string marker;
    return ListMarker(line, marker) > 0;
}

U64 HeadingLevel(const std::string& line) {
    U64 hashes = 0;
    while (hashes < line.size() && line[hashes] == '#') {
        ++hashes;
    }
    if (hashes == 0 || hashes >= line.size() || line[hashes] != ' ') {
        return 0;
    }
    return hashes;
}

bool IsQuoteLine(const std::string& line) {
    return !line.empty() && line[0] == '>';
}

std::string QuoteContent(const std::string& line) {
    U64 s = 1;
    while (s < line.size() && line[s] == ' ') {
        ++s;
    }
    return line.substr(s);
}

bool ParseCalloutTag(const std::string& line, StyleId& tone) {
    for (U64 i = 0; i < kCalloutTags.size(); ++i) {
        if (line == kCalloutTags[i]) {
            tone = static_cast<StyleId>(i);
            return true;
        }
    }
    return false;
}

void TintStyles(std::vector<StyleId>& styles, StyleId tone) {
    for (auto& style : styles) {
        if (style <= Style::BoldItalic) {
            style = Style::Callout(tone, style);
        }
    }
}

std::string TrimRight(std::string s) {
    while (!s.empty() && (s.back() == '\r' || s.back() == ' ' || s.back() == '\t')) {
        s.pop_back();
    }
    return s;
}

std::string TrimCell(const std::string& s) {
    const U64 begin = s.find_first_not_of(" \t");
    if (begin == std::string::npos) {
        return "";
    }
    const U64 end = s.find_last_not_of(" \t");
    return s.substr(begin, end - begin + 1);
}

std::vector<std::string> SplitTableRow(const std::string& line) {
    std::vector<std::string> cells;
    std::string current;
    bool pipe = false;
    for (U64 i = 0; i < line.size(); ++i) {
        const char c = line[i];
        if (c == '\\') {
            U64 run = 1;
            while (i + run < line.size() && line[i + run] == '\\') {
                ++run;
            }
            const bool escapesPipe = (run % 2 == 1) && i + run < line.size() && line[i + run] == '|';
            current.append(escapesPipe ? run - 1 : run, '\\');
            if (escapesPipe) {
                current += '|';
                ++run;
            }
            i += run - 1;
            continue;
        }
        if (c == '`') {
            const U64 close = line.find('`', i + 1);
            if (close != std::string::npos) {
                for (U64 k = i; k <= close; ++k) {
                    if (line[k] == '\\' && k + 1 < close && line[k + 1] == '|') {
                        current += '|';
                        ++k;
                    } else {
                        current += line[k];
                    }
                }
                i = close;
                continue;
            }
        }
        if (c == '|') {
            pipe = true;
            cells.push_back(std::move(current));
            current.clear();
            continue;
        }
        current += c;
    }
    cells.push_back(std::move(current));
    if (!pipe) {
        return {};
    }
    const auto blank = [](const std::string& s) {
        return s.find_first_not_of(" \t") == std::string::npos;
    };
    if (!cells.empty() && blank(cells.front())) {
        cells.erase(cells.begin());
    }
    if (!cells.empty() && blank(cells.back())) {
        cells.pop_back();
    }
    for (auto& cell : cells) {
        cell = TrimCell(cell);
    }
    return cells;
}

bool ParseDelimiterRow(const std::string& line, std::vector<I8>& aligns) {
    const auto cells = SplitTableRow(line);
    if (cells.empty()) {
        return false;
    }
    aligns.clear();
    for (const auto& cell : cells) {
        if (cell.empty()) {
            return false;
        }
        const bool left = cell.front() == ':';
        const bool right = cell.size() > 1 && cell.back() == ':';
        const U64 begin = left ? 1 : 0;
        const U64 end = cell.size() - (right ? 1 : 0);
        if (end <= begin) {
            return false;
        }
        for (U64 i = begin; i < end; ++i) {
            if (cell[i] != '-') {
                return false;
            }
        }
        aligns.push_back(left && right ? 0 : right ? 1 : -1);
    }
    return true;
}

bool StartsBlock(const std::string& line) {
    return IsFence(line) || IsRule(line) || HeadingLevel(line) > 0 || IsQuoteLine(line) || IsListItem(line);
}

void ParseInline(const std::string& line, StyleId baseStyle, std::string& display, std::vector<StyleId>& styles,
                 std::vector<LinkSpan>& links) {
    const U64 n = line.size();
    const auto emit = [&](char c, StyleId s) {
        display += c;
        styles.push_back(s);
    };
    const auto emitRange = [&](U64 a, U64 b, StyleId s) {
        for (U64 k = a; k < b; ++k) {
            emit(line[k], s);
        }
    };
    const auto delimited = [&](U64 i, std::string_view open, StyleId s) -> U64 {
        if (line.compare(i, open.size(), open) != 0) {
            return 0;
        }
        const U64 j = line.find(open, i + open.size());
        if (j == std::string::npos) {
            return 0;
        }
        emitRange(i + open.size(), j, s);
        return j + open.size();
    };

    U64 i = 0;
    while (i < n) {
        const char c = line[i];
        U64 resume = 0;
        for (const auto& [open, style] : kInlineDelimiters) {
            resume = delimited(i, open, style);
            if (resume != 0) {
                break;
            }
        }
        if (resume != 0) {
            i = resume;
            continue;
        }
        if (c == '*' || c == '_') {
            const U64 j = line.find(c, i + 1);
            if (j != std::string::npos && j > i + 1) {
                emitRange(i + 1, j, Style::Italic);
                i = j + 1;
                continue;
            }
        }
        if (c == '[') {
            const U64 close = line.find(']', i + 1);
            if (close != std::string::npos && close + 1 < n && line[close + 1] == '(') {
                const U64 paren = line.find(')', close + 2);
                if (paren != std::string::npos) {
                    const U64 startCol = display.size();
                    emitRange(i + 1, close, Style::Link);
                    links.push_back({startCol, display.size(), line.substr(close + 2, paren - (close + 2))});
                    i = paren + 1;
                    continue;
                }
            }
        }
        emit(c, baseStyle);
        ++i;
    }
}

struct BlockScanner {
    const std::vector<std::string>& source;
    F32 body;
    F32 lineHeight;
    F32 indentUnit;
    std::vector<Block> blocks;
    bool prevHeading = false;
    bool prevList = false;
    U64 index = 0;

    BlockScanner(const std::vector<std::string>& source, F32 body)
        : source(source), body(body), lineHeight(body * kLineHeightRatio), indentUnit(body * kIndentEmRatio) {}

    std::vector<Block> run() {
        while (index < source.size()) {
            const std::string line = TrimRight(source[index]);
            if (line.empty()) {
                ++index;
                continue;
            }
            if (scanFencedCode(line) || scanRule(line) || scanHeading(line) || scanQuote(line) ||
                scanTable(line) || scanListItem(line)) {
                continue;
            }
            scanParagraph();
        }
        return std::move(blocks);
    }

    bool tableStartsAt(U64 at, std::vector<std::string>& header, std::vector<I8>& aligns) const {
        if (at + 1 >= source.size()) {
            return false;
        }
        const std::string line = TrimRight(source[at]);
        if (line.find('|') == std::string::npos) {
            return false;
        }
        header = SplitTableRow(line);
        return !header.empty() && ParseDelimiterRow(TrimRight(source[at + 1]), aligns) &&
               aligns.size() == header.size();
    }

    bool tableStartsAt(U64 at) const {
        std::vector<std::string> header;
        std::vector<I8> aligns;
        return tableStartsAt(at, header, aligns);
    }

    F32 gapAbove(F32 ratio) const {
        return blocks.empty() ? 0.0f : lineHeight * (prevHeading ? kAfterHeadingGapRatio : ratio);
    }

    F32 boxPad(Block::Kind kind) const {
        switch (kind) {
            case Block::Kind::Code: return body * kCodePadRatio;
            case Block::Kind::Table: return body * kTableRowPadRatio * 0.5f;
            case Block::Kind::Callout: return body * kCalloutPadEmRatio;
            default: return 0.0f;
        }
    }

    F32 padBelowPrevious() const {
        return blocks.empty() ? 0.0f : boxPad(blocks.back().kind);
    }

    F32 topGapFor(Block::Kind kind, F32 rowGap) const {
        return rowGap + padBelowPrevious() + boxPad(kind);
    }

    void emit(Block block, bool heading = false, bool list = false) {
        blocks.push_back(std::move(block));
        prevHeading = heading;
        prevList = list;
    }

    bool scanFencedCode(const std::string& line) {
        if (!IsFence(line)) {
            return false;
        }
        const auto syntax = SyntaxHighlighter::LanguageForTag(FenceTag(line));
        std::vector<std::string> code;
        ++index;
        while (index < source.size() && !IsFence(source[index])) {
            code.push_back(source[index]);
            ++index;
        }
        if (index < source.size()) {
            ++index;
        }
        Block block;
        block.kind = Block::Kind::Code;
        block.text = JoinLines(code);
        block.fontSize = body;
        block.topGap = topGapFor(block.kind, gapAbove(kParagraphGapRatio));
        block.baseStyle = Style::CodeBlock;
        block.syntax = syntax;
        emit(std::move(block));
        return true;
    }

    bool scanRule(const std::string& line) {
        if (!IsRule(line)) {
            return false;
        }
        Block block;
        block.kind = Block::Kind::Rule;
        block.fontSize = body;
        block.topGap = topGapFor(block.kind, gapAbove(kParagraphGapRatio));
        emit(std::move(block));
        ++index;
        return true;
    }

    bool scanHeading(const std::string& line) {
        const U64 level = HeadingLevel(line);
        if (level == 0) {
            return false;
        }
        const U64 scale = std::min<U64>(level, 3);
        Block block;
        block.kind = Block::Kind::Heading;
        block.text = line.substr(level + 1);
        block.fontSize = body * Typography::HeadingScale[scale - 1];
        block.topGap = topGapFor(block.kind, gapAbove(kHeadingGapRatio));
        block.baseStyle = Style::Bold;
        emit(std::move(block), true);
        ++index;
        return true;
    }

    bool scanQuote(const std::string& line) {
        if (!IsQuoteLine(line)) {
            return false;
        }
        std::vector<std::string> quote;
        while (index < source.size()) {
            const std::string current = TrimRight(source[index]);
            if (!IsQuoteLine(current)) {
                break;
            }
            quote.push_back(QuoteContent(current));
            ++index;
        }
        Block block;
        block.fontSize = body;
        StyleId tone = 0;
        if (ParseCalloutTag(quote.front(), tone)) {
            block.kind = Block::Kind::Callout;
            block.tone = tone;
            block.text = JoinLines({quote.begin() + 1, quote.end()});
            block.indent = body * kCalloutPadEmRatio;
        } else {
            block.kind = Block::Kind::Quote;
            block.text = JoinLines(quote);
            block.indent = body * kQuoteIndentEmRatio;
        }
        block.topGap = topGapFor(block.kind, gapAbove(kParagraphGapRatio));
        emit(std::move(block));
        return true;
    }

    bool scanTable(const std::string&) {
        std::vector<std::string> header;
        std::vector<I8> aligns;
        if (!tableStartsAt(index, header, aligns)) {
            return false;
        }
        Block block;
        block.kind = Block::Kind::Table;
        block.fontSize = body;
        block.topGap = topGapFor(block.kind, gapAbove(kParagraphGapRatio));
        block.table.aligns = std::move(aligns);
        block.table.rows.push_back(std::move(header));
        index += 2;
        while (index < source.size()) {
            const std::string row = TrimRight(source[index]);
            if (row.empty() || row.find('|') == std::string::npos || StartsBlock(row)) {
                break;
            }
            auto cells = SplitTableRow(row);
            if (cells.empty()) {
                break;
            }
            cells.resize(block.table.aligns.size());
            block.table.rows.push_back(std::move(cells));
            ++index;
        }
        emit(std::move(block));
        return true;
    }

    bool scanListItem(const std::string& line) {
        std::string marker;
        const U64 contentStart = ListMarker(line, marker);
        if (contentStart == 0) {
            return false;
        }
        Block block;
        block.kind = Block::Kind::ListItem;
        block.text = line.substr(contentStart);
        block.fontSize = body;
        block.topGap = topGapFor(block.kind, lineHeight * (prevList ? kListGapRatio
                                                         : prevHeading ? kAfterHeadingGapRatio
                                                                       : kParagraphGapRatio));
        block.indent = indentUnit * static_cast<F32>(LeadingSpaces(line) / 2 + 1);
        block.marker = std::move(marker);
        emit(std::move(block), false, true);
        ++index;
        return true;
    }

    void scanParagraph() {
        std::vector<std::string> paragraph;
        while (index < source.size()) {
            const std::string current = TrimRight(source[index]);
            if (current.empty() || StartsBlock(current) || tableStartsAt(index)) {
                break;
            }
            paragraph.push_back(current);
            ++index;
        }
        Block block;
        block.kind = Block::Kind::Paragraph;
        block.text = JoinLines(paragraph);
        block.fontSize = body;
        block.topGap = topGapFor(block.kind, gapAbove(kParagraphGapRatio));
        emit(std::move(block));
    }
};

struct Document {
    std::vector<F32> lineScale;
    std::vector<F32> lineTopGap;
    std::vector<F32> lineIndent;
    std::vector<U8> lineSameRow;
    std::vector<F32> lineWrapWidth;
    std::vector<F32> lineRightInset;
    std::vector<U8> lineGutterDigits;
    std::vector<std::vector<StyleId>> styles;
    std::vector<std::vector<LinkSpan>> links;
    std::vector<Deco> decos;
    std::vector<Table> tables;
    std::string plainValue;
    F32 trailingPad = 0.0f;

    F32 scaleAt(U64 line) const {
        return line < lineScale.size() ? lineScale[line] : 1.0f;
    }

    const std::string* linkAt(TextGrid::Position pos) const {
        if (pos.line >= links.size()) {
            return nullptr;
        }
        for (const auto& span : links[pos.line]) {
            if (pos.column >= span.start && pos.column < span.end) {
                return &span.url;
            }
        }
        return nullptr;
    }
};

struct DocumentBuilder {
    F32 body;
    F32 codeIndent;
    F32 codeRightInset;
    SyntaxHighlighter& highlighter;
    Document document;
    std::vector<std::string> lines;

    DocumentBuilder(F32 body, F32 codeIndent, F32 codeRightInset, SyntaxHighlighter& highlighter)
        : body(body), codeIndent(codeIndent), codeRightInset(codeRightInset), highlighter(highlighter) {}

    Document build(const std::vector<Block>& blocks) {
        for (const auto& block : blocks) {
            switch (block.kind) {
                case Block::Kind::Rule: emitRule(block); break;
                case Block::Kind::Table: emitTable(block); break;
                case Block::Kind::Callout: emitCallout(block); break;
                default: emitTextBlock(block); break;
            }
        }
        if (lines.empty()) {
            addLine("", 1.0f, 0.0f, 0.0f, {}, {});
        }
        if (!blocks.empty()) {
            document.trailingPad = trailingPadFor(blocks.back().kind);
        }
        document.plainValue = JoinLines(lines);
        return std::move(document);
    }

    F32 trailingPadFor(Block::Kind kind) const {
        switch (kind) {
            case Block::Kind::Code: return body * kCodePadRatio;
            case Block::Kind::Table: return body * kTableRowPadRatio * 0.5f;
            case Block::Kind::Callout: return body * kCalloutPadEmRatio;
            default: return 0.0f;
        }
    }

    void emitCallout(const Block& block) {
        const U64 firstLine = lines.size();
        const auto addCalloutLine = [&](const std::string& source, StyleId baseStyle, F32 scale, F32 gap,
                                        bool tint) {
            std::vector<StyleId> styles;
            std::vector<LinkSpan> links;
            std::string display;
            ParseInline(source, baseStyle, display, styles, links);
            if (tint) {
                TintStyles(styles, block.tone);
            }
            addLine(std::move(display), scale, gap, block.indent, std::move(styles), std::move(links), false,
                    block.indent);
        };
        addCalloutLine(std::string(kCalloutIcons[block.tone]) + "  " + kCalloutTitles[block.tone], Style::Bold,
                       kCalloutTitleScale, block.topGap, true);
        if (!block.text.empty()) {
            F32 gap = body * kCalloutTitleGapEmRatio;
            for (const auto& source : SplitLines(block.text)) {
                addCalloutLine(source, Style::Plain, 1.0f, gap, true);
                gap = 0.0f;
            }
        }
        document.decos.push_back({.kind = Deco::Kind::Callout, .first = firstLine, .last = lines.size() - 1,
                                  .tone = block.tone});
    }

    void addLine(std::string text, F32 scale, F32 gap, F32 indent,
                 std::vector<StyleId> lineStyles, std::vector<LinkSpan> lineLinks, bool sameRow = false,
                 F32 rightInset = 0.0f) {
        lines.push_back(std::move(text));
        document.lineScale.push_back(scale);
        document.lineTopGap.push_back(gap);
        document.lineIndent.push_back(indent);
        document.lineSameRow.push_back(sameRow ? 1 : 0);
        document.lineWrapWidth.push_back(0.0f);
        document.lineRightInset.push_back(rightInset);
        document.lineGutterDigits.push_back(0);
        document.styles.push_back(std::move(lineStyles));
        document.links.push_back(std::move(lineLinks));
    }

    void emitTable(const Block& block) {
        const F32 rowPad = body * kTableRowPadRatio;
        Table table;
        table.columns = block.table.aligns.size();
        table.rows = block.table.rows.size();
        table.aligns = block.table.aligns;
        for (U64 r = 0; r < table.rows; ++r) {
            for (U64 c = 0; c < table.columns; ++c) {
                TableCell cell;
                std::vector<StyleId> styles;
                std::vector<LinkSpan> links;
                ParseInline(block.table.rows[r][c], r == 0 ? Style::Bold : Style::Plain, cell.text, styles, links);
                cell.styles = styles;
                cell.line = lines.size();
                const F32 gap = c > 0 ? 0.0f : r == 0 ? block.topGap + rowPad * 0.5f : rowPad;
                addLine(cell.text, 1.0f, gap, 0.0f, std::move(styles), std::move(links), c > 0);
                table.cells.push_back(std::move(cell));
            }
        }
        document.decos.push_back({.kind = Deco::Kind::Table, .table = document.tables.size()});
        document.tables.push_back(std::move(table));
    }

    void emitRule(const Block& block) {
        const U64 line = lines.size();
        addLine("", kRuleRowScale, block.topGap, 0.0f, {}, {});
        document.decos.push_back({.kind = Deco::Kind::Rule, .first = line, .last = line});
    }

    void emitTextBlock(const Block& block) {
        const F32 scale = block.fontSize / body;
        const F32 indent = block.indent + (block.kind == Block::Kind::Code ? codeIndent : 0.0f);
        const U64 firstLine = lines.size();

        const auto sources = SplitLines(block.text);
        const auto syntaxStyles = block.syntax.has_value() ? highlighter.highlight(sources, *block.syntax)
                                                           : std::vector<std::vector<StyleId>>{};
        for (U64 li = 0; li < sources.size(); ++li) {
            const F32 gap = li == 0 ? block.topGap : 0.0f;
            std::vector<StyleId> styles;
            std::vector<LinkSpan> links;
            std::string display;
            if (block.inlineParse()) {
                ParseInline(sources[li], block.baseStyle, display, styles, links);
            } else {
                display = sources[li];
                styles.assign(display.size(), block.baseStyle);
                if (li < syntaxStyles.size()) {
                    for (U64 column = 0; column < styles.size() && column < syntaxStyles[li].size(); ++column) {
                        if (syntaxStyles[li][column] != SyntaxHighlighter::Default) {
                            styles[column] = Style::Syntax(syntaxStyles[li][column]);
                        }
                    }
                }
            }
            addLine(std::move(display), scale, gap, indent, std::move(styles), std::move(links), false,
                    block.kind == Block::Kind::Code ? codeRightInset : 0.0f);
        }
        const U64 lastLine = lines.size() - 1;

        switch (block.kind) {
            case Block::Kind::Code: {
                const U8 digits = std::max(kCodeMinLineDigits,
                                           static_cast<U8>(std::to_string(lastLine - firstLine + 1).size()));
                std::fill(document.lineGutterDigits.begin() + firstLine, document.lineGutterDigits.end(), digits);
                document.decos.push_back({.kind = Deco::Kind::Code, .first = firstLine, .last = lastLine,
                                          .indent = indent, .digits = digits});
                break;
            }
            case Block::Kind::Quote:
                document.decos.push_back({.kind = Deco::Kind::Quote, .first = firstLine, .last = lastLine, .indent = indent});
                break;
            case Block::Kind::ListItem:
                document.decos.push_back({.kind = Deco::Kind::ListItem, .first = firstLine, .last = firstLine,
                                          .marker = block.marker, .indent = indent});
                break;
            default:
                break;
        }
    }
};

}  // namespace

struct TextMarkdown::Impl {
    struct DecorationFrame {
        Rect rect;
        std::optional<Rect> clip;
        F32 body = 0.0f;
        F32 bodyPad = 0.0f;
        F32 width = 0.0f;
        bool on = false;
        ColorRGBA<F32> codeBackground;
        ColorRGBA<F32> codeBlock;
        ColorRGBA<F32> codeBlockBorder;
        F32 codeBorder = 0.0f;
        F32 codeGlyph = 0.0f;
        F32 lineNumberAdvance = 0.0f;
        ColorRGBA<F32> bar;
        ColorRGBA<F32> rule;
        ColorRGBA<F32> marker;
        std::array<ColorRGBA<F32>, kCalloutTones> callout;
        F32 scrollX = 0.0f;
        const TextGrid::Metrics* metrics = nullptr;

        F32 lineTop(U64 line) const {
            return line < metrics->sourceLines.size() ? metrics->sourceLines[line].top : rect.y;
        }
        F32 lineHeight(U64 line) const {
            return line < metrics->sourceLines.size() ? metrics->sourceLines[line].height : 0.0f;
        }
        F32 lineBottom(U64 line) const { return lineTop(line) + lineHeight(line); }
    };

    struct DecorationPools {
        std::vector<Box::Instance> boxes;
        std::vector<Box::Instance> tableBoxes;
        std::vector<Box::Instance> codeBoxes;
        std::array<std::vector<Box::Instance>, kCalloutTones> callouts;
        std::vector<Label::Instance> markers;

        bool full() const {
            return boxes.size() >= kMaxDecorations || tableBoxes.size() >= kMaxDecorations ||
                   codeBoxes.size() >= kMaxDecorations ||
                   markers.size() >= kMaxDecorations;
        }
    };

    Config config;
    TextGrid grid;
    std::array<Box, kCalloutTones> calloutChrome;
    Box codeChrome;
    Box tableChrome;
    Box decorations;
    Label markers;
    SyntaxHighlighter highlighter;
    Document document;
    mutable TextMetrics textMetrics;
    U64 styleRevision = 0;
    bool parsed = false;
    bool tablesLayoutValid = false;
    F32 tablesLayoutWidth = -1.0f;
    TextGrid::WidthLayout tablesLayout;

    void rebuild() {
        ++styleRevision;
        const std::vector<Block> blocks = BlockScanner(SplitLines(config.value), config.fontSize).run();
        const Padding padding = gridPadding();
        const F32 inset = codeInset();
        document = DocumentBuilder(config.fontSize, std::max(0.0f, inset - padding.left),
                                   std::max(0.0f, inset - padding.right), highlighter).build(blocks);
        tablesLayoutValid = false;
    }

    F32 codeGlyphSize() const {
        return config.fontSize * styleScale(Style::CodeBlock);
    }

    F32 lineNumberAdvance() const {
        const F32 glyph = codeGlyphSize();
        const F32 measured = textMetrics.measure(kCodeFont, "0", glyph);
        return measured > 0.0f ? measured : glyph * kFallbackGlyphWidthRatio;
    }

    std::vector<F32> lineIndents() const {
        std::vector<F32> indents = document.lineIndent;
        const F32 advance = lineNumberAdvance();
        for (U64 i = 0; i < indents.size() && i < document.lineGutterDigits.size(); ++i) {
            if (document.lineGutterDigits[i] > 0) {
                indents[i] += (static_cast<F32>(document.lineGutterDigits[i]) + kCodeGutterGapChars) * advance;
            }
        }
        return indents;
    }

    F32 minimumContentWidth(const std::vector<F32>& indents) const {
        const F32 glyph = glyphMinimumWidth();
        F32 widest = widestTableWidth();
        for (U64 i = 0; i < indents.size(); ++i) {
            if (i < document.lineWrapWidth.size() && document.lineWrapWidth[i] > 0.0f) {
                continue;
            }
            const F32 rightInset = i < document.lineRightInset.size() ? document.lineRightInset[i] : 0.0f;
            if (indents[i] <= 0.0f && rightInset <= 0.0f) {
                continue;
            }
            widest = std::max(widest, indents[i] + glyph * document.scaleAt(i) + rightInset);
        }
        return widest;
    }

    F32 codeInset() const {
        const F32 capHeight = config.fontSize * styleScale(Style::CodeBlock) * kCodeCapHeightRatio;
        const F32 leading = config.fontSize * kLineHeightRatio - capHeight;
        return config.fontSize * kCodePadRatio + leading * 0.5f;
    }

    static StyleId configStyle(StyleId id) {
        return id >= Style::SyntaxBase ? Style::CodeBlock : id;
    }

    F32 styleScale(StyleId style) const {
        const StyleId id = configStyle(style);
        if (id == 0 || id > config.styleScales.size() || config.styleScales[id - 1] <= 0.0f) {
            return 1.0f;
        }
        return config.styleScales[id - 1];
    }

    std::string styleFont(StyleId style) const {
        const StyleId id = configStyle(style);
        if (id == 0 || id > config.styleFonts.size() || config.styleFonts[id - 1].empty()) {
            return kBodyFont;
        }
        return config.styleFonts[id - 1];
    }

    bool styleHasBackground(StyleId id) const {
        return id != 0 && id <= config.styleBackgroundColorKeys.size() &&
               !config.styleBackgroundColorKeys[id - 1].empty();
    }

    bool fontsReady() const {
        const F32 body = config.fontSize;
        if (textMetrics.measure(kBodyFont, "0", body) <= 0.0f) {
            return false;
        }
        for (const auto& font : config.styleFonts) {
            if (!font.empty() && textMetrics.measure(font, "0", body) <= 0.0f) {
                return false;
            }
        }
        return true;
    }

    F32 cellNaturalWidth(const TableCell& cell) const {
        const F32 body = config.fontSize;
        F32 width = 0.0f;
        U64 i = 0;
        while (i < cell.text.size()) {
            const StyleId style = i < cell.styles.size() ? cell.styles[i] : Style::Plain;
            U64 j = i + 1;
            while (j < cell.text.size() && (j < cell.styles.size() ? cell.styles[j] : Style::Plain) == style) {
                ++j;
            }
            const F32 glyphSize = body * styleScale(style);
            width += textMetrics.measure(styleFont(style), cell.text.substr(i, j - i), glyphSize);
            if (styleHasBackground(style)) {
                width += 2.0f * glyphSize * Typography::StyleBackgroundPadRatio;
            }
            i = j;
        }
        return width;
    }

    F32 widestTableWidth() const {
        F32 widest = 0.0f;
        for (const auto& table : document.tables) {
            widest = std::max(widest, table.width);
        }
        return widest;
    }

    TextGrid::WidthLayout layoutForTextWidth(F32 textWidth) {
        if (!(tablesLayoutValid && tablesLayoutWidth == textWidth)) {
            layoutTables(textWidth);
            std::vector<F32> indents = lineIndents();
            const F32 minimumWidth = minimumContentWidth(indents);
            tablesLayout = {
                .lineIndent = std::move(indents),
                .lineWrapWidth = document.lineWrapWidth,
                .contentMinWidth = minimumWidth,
            };
            tablesLayoutValid = fontsReady();
            tablesLayoutWidth = textWidth;
        }
        return tablesLayout;
    }

    F32 glyphMinimumWidth() const {
        const F32 measured = textMetrics.measure(kBodyFont, "W", config.fontSize);
        return measured > 0.0f ? measured : config.fontSize * kFallbackGlyphWidthRatio;
    }

    void layoutTables(F32 textWidth) {
        const F32 body = config.fontSize;
        const bool bounded = std::isfinite(textWidth) && textWidth > 0.0f;
        const F32 available = bounded ? std::max(1.0f, textWidth) : std::numeric_limits<F32>::max();
        const F32 glyphMinimum = glyphMinimumWidth();

        for (auto& table : document.tables) {
            if (table.columns == 0 || table.rows == 0) {
                continue;
            }
            const F32 slot = available / static_cast<F32>(table.columns);
            const F32 padX = std::min(body * kTableCellPadXRatio, slot * kTableNarrowPadShare);
            const F32 minColumn = std::max(glyphMinimum,
                                           std::min(body * kTableMinColumnEm, slot - 2.0f * padX));
            std::vector<F32> columnWidths(table.columns, minColumn);
            for (U64 i = 0; i < table.cells.size(); ++i) {
                table.cells[i].width = cellNaturalWidth(table.cells[i]);
                columnWidths[i % table.columns] = std::max(columnWidths[i % table.columns], table.cells[i].width);
            }
            shrinkColumns(columnWidths, available - 2.0f * padX * static_cast<F32>(table.columns), minColumn);

            table.columnX.assign(table.columns + 1, 0.0f);
            for (U64 c = 0; c < table.columns; ++c) {
                table.columnX[c + 1] = table.columnX[c] + columnWidths[c] + 2.0f * padX;
            }
            if (bounded && table.columnX[table.columns] > available &&
                table.columnX[table.columns] - available <= kWidthTolerance) {
                table.columnX[table.columns] = available;
            }
            table.width = table.columnX[table.columns];

            for (U64 i = 0; i < table.cells.size(); ++i) {
                const auto& cell = table.cells[i];
                const U64 c = i % table.columns;
                const F32 columnWidth = columnWidths[c];
                F32 shift = 0.0f;
                if (cell.width <= columnWidth) {
                    if (table.aligns[c] == 0) {
                        shift = (columnWidth - cell.width) * 0.5f;
                    } else if (table.aligns[c] > 0) {
                        shift = columnWidth - cell.width;
                    }
                }
                document.lineIndent[cell.line] = table.columnX[c] + padX + shift;
                document.lineWrapWidth[cell.line] = columnWidth + kTableWrapSlack;
            }
        }
    }

    std::vector<std::string> gridStyleColorKeys() const {
        auto keys = config.styleColorKeys;
        keys.resize(Style::CalloutBase - 1);
        for (StyleId tone = 0; tone < kCalloutTones; ++tone) {
            for (StyleId variant = 0; variant < Style::CalloutVariants; ++variant) {
                keys.emplace_back(kCalloutColorKeys[tone]);
            }
        }
        for (StyleId syntax = 0; syntax < Style::SyntaxStyles; ++syntax) {
            keys.push_back(syntax < config.syntaxColorKeys.size() ? config.syntaxColorKeys[syntax] : "");
        }
        return keys;
    }

    std::vector<F32> gridStyleScales() const {
        auto scales = config.styleScales;
        scales.resize(Style::SyntaxBase - 1, 0.0f);
        scales.resize(Style::SyntaxBase - 1 + Style::SyntaxStyles, styleScale(Style::CodeBlock));
        return scales;
    }

    std::vector<std::string> gridStyleFonts() const {
        auto fonts = config.styleFonts;
        fonts.resize(Style::CalloutBase - 1);
        for (StyleId tone = 0; tone < kCalloutTones; ++tone) {
            fonts.emplace_back("");
            for (StyleId emphasis = Style::Bold; emphasis <= Style::BoldItalic; ++emphasis) {
                fonts.push_back(config.styleFonts.size() >= emphasis ? config.styleFonts[emphasis - 1] : "");
            }
        }
        fonts.resize(Style::SyntaxBase - 1 + Style::SyntaxStyles, styleFont(Style::CodeBlock));
        return fonts;
    }

    static void shrinkColumns(std::vector<F32>& widths, F32 available, F32 minColumn) {
        F32 total = 0.0f;
        F32 widest = minColumn;
        for (const F32 w : widths) {
            total += w;
            widest = std::max(widest, w);
        }
        if (total <= available) {
            return;
        }
        F32 lo = minColumn;
        F32 hi = widest;
        for (U64 iteration = 0; iteration < kTableShrinkIterations; ++iteration) {
            const F32 mid = 0.5f * (lo + hi);
            F32 sum = 0.0f;
            for (const F32 w : widths) {
                sum += std::min(w, mid);
            }
            (sum > available ? hi : lo) = mid;
        }
        for (auto& w : widths) {
            w = std::min(w, lo);
        }
    }

    F32 rowHeight(U64 line) const {
        return config.fontSize * document.scaleAt(line) * kLineHeightRatio;
    }

    Padding gridPadding() const {
        const F32 horizontal = config.fontSize * Typography::TextPaddingRatio;
        Padding padding = config.padding.value_or(Padding{horizontal, 0.0f, horizontal, 0.0f});
        padding.bottom += document.trailingPad;
        return padding;
    }

    TextGrid::Config buildGridConfig() {
        std::vector<F32> indents = lineIndents();
        const F32 minimumWidth = minimumContentWidth(indents);
        return {
            .id = config.id + ":grid",
            .value = document.plainValue,
            .editable = false,
            .fontSize = config.fontSize,
            .lineHeight = Typography::BodyLineHeight,
            .fontName = kBodyFont,
            .monospace = false,
            .showActiveLine = false,
            .scrollbar = config.scrollbar,
            .wrap = TextGrid::Wrap::Word,
            .padding = gridPadding(),
            .lineScale = document.lineScale,
            .lineTopGap = document.lineTopGap,
            .lineIndent = std::move(indents),
            .lineSameRow = document.lineSameRow,
            .lineWrapWidth = document.lineWrapWidth,
            .lineRightInset = document.lineRightInset,
            .contentMinWidth = minimumWidth,
            .backgroundColorKey = config.backgroundColorKey,
            .textColorKey = config.textColorKey,
            .lineNumberColorKey = config.lineNumberColorKey,
            .gutterSeparatorColorKey = config.gutterSeparatorColorKey,
            .selectionColorKey = config.selectionColorKey,
            .selectionMatchColorKey = config.selectionMatchColorKey,
            .activeLineColorKey = config.activeLineColorKey,
            .cursorColorKey = config.cursorColorKey,
            .scrollbarTrackColorKey = config.scrollbarTrackColorKey,
            .scrollbarThumbColorKey = config.scrollbarThumbColorKey,
            .styleColorKeys = gridStyleColorKeys(),
            .styleFonts = gridStyleFonts(),
            .styleBackgroundColorKeys = config.styleBackgroundColorKeys,
            .styleScales = gridStyleScales(),
            .maxLineSegments = kGridLineSegments,
            .styleRevision = styleRevision,
            .styler = [this](const std::vector<std::string>&, U64)
                          -> const std::vector<std::vector<StyleId>>& {
                return document.styles;
            },
            .widthLayout = document.tables.empty()
                ? std::function<TextGrid::WidthLayout(F32)>{}
                : [this](F32 textWidth) { return layoutForTextWidth(textWidth); },
            .onPositionClick = [this](TextGrid::Position pos) -> bool {
                if (const std::string* url = document.linkAt(pos)) {
                    (void)Platform::OpenUrl(*url);
                    return true;
                }
                return false;
            },
            .isPositionInteractive = [this](TextGrid::Position pos) -> bool {
                return document.linkAt(pos) != nullptr;
            },
        };
    }

    DecorationFrame decorationFrame(const Context& ctx, const Rect& rect, const Rect& clipPixel) const {
        const auto& metrics = grid.metrics();
        const F32 body = config.fontSize;
        return {
            .rect = rect,
            .clip = clipPixel,
            .body = body,
            .bodyPad = metrics.padding.left,
            .width = std::max(0.0f, rect.width - metrics.scrollbarGutter),
            .on = !rect.empty(),
            .codeBackground = ctx.color(config.scrollbarTrackColorKey),
            .codeBlock = ctx.color(config.codeBlockColorKey),
            .codeBlockBorder = ctx.color(config.codeBlockBorderColorKey),
            .codeBorder = std::max(1.0f, std::round(body * kCodeBorderRatio * 2.0f) * 0.5f),
            .codeGlyph = codeGlyphSize(),
            .lineNumberAdvance = lineNumberAdvance(),
            .bar = ctx.color(config.lineNumberColorKey),
            .rule = ctx.color(config.gutterSeparatorColorKey),
            .marker = ctx.color(config.lineNumberColorKey),
            .callout = calloutColors(ctx),
            .scrollX = metrics.scrollX,
            .metrics = &metrics,
        };
    }

    static std::array<ColorRGBA<F32>, kCalloutTones> calloutColors(const Context& ctx) {
        std::array<ColorRGBA<F32>, kCalloutTones> colors;
        for (U64 tone = 0; tone < kCalloutTones; ++tone) {
            colors[tone] = ctx.color(kCalloutColorKeys[tone]);
        }
        return colors;
    }

    void addCallout(const DecorationFrame& frame, const Deco& deco, DecorationPools& pools) const {
        auto& instances = pools.callouts[deco.tone];
        if (instances.size() >= kMaxCallouts) {
            return;
        }
        const F32 pad = frame.body * kCalloutPadEmRatio;
        const F32 top = frame.lineTop(deco.first) - pad;
        const F32 bottom = frame.lineBottom(deco.last) + pad;
        if (frame.clip.has_value() && (bottom <= frame.clip->y || top >= frame.clip->bottom())) {
            return;
        }
        auto fill = frame.callout[deco.tone];
        fill.a *= kCalloutFillAlpha;
        instances.push_back({
            .rect = {frame.rect.x, top, frame.width, bottom - top},
            .visible = frame.on,
            .backgroundColor = fill,
        });
    }

    void addRule(const DecorationFrame& frame, const Deco& deco, DecorationPools& pools) const {
        const F32 thickness = std::max(1.0f, frame.body * kRuleThicknessRatio);
        const F32 center = frame.lineTop(deco.first) + rowHeight(deco.first) * 0.5f;
        pools.boxes.push_back({
            .rect = {frame.rect.x, center - thickness * 0.5f, frame.width, thickness},
            .visible = frame.on,
            .backgroundColor = frame.rule,
        });
    }

    void addCodeBlock(const DecorationFrame& frame, const Deco& deco, DecorationPools& pools) const {
        const F32 pad = frame.body * kCodePadRatio;
        const F32 top = frame.lineTop(deco.first);
        const F32 bottom = frame.lineBottom(deco.last);
        pools.codeBoxes.push_back({
            .rect = {frame.rect.x, top - pad, frame.width, (bottom - top) + 2.0f * pad},
            .visible = frame.on,
            .backgroundColor = frame.codeBlock,
        });

        const F32 numbersLeft = frame.rect.x + frame.bodyPad + deco.indent - frame.scrollX;
        const F32 numbersWidth = static_cast<F32>(deco.digits) * frame.lineNumberAdvance;
        const F32 separatorCenter = numbersLeft + numbersWidth +
                                    kCodeGutterGapChars * 0.5f * frame.lineNumberAdvance;
        pools.boxes.push_back({
            .rect = {std::round(separatorCenter - frame.codeBorder * 0.5f), top - pad + frame.codeBorder,
                     frame.codeBorder, (bottom - top) + 2.0f * (pad - frame.codeBorder)},
            .visible = frame.on,
            .backgroundColor = frame.rule,
        });

        for (U64 line = deco.first; line <= deco.last && pools.markers.size() < kMaxDecorations; ++line) {
            const F32 lineTop = frame.lineTop(line);
            const F32 height = rowHeight(line);
            if (frame.clip.has_value() && lineTop >= frame.clip->bottom()) {
                break;
            }
            if (frame.clip.has_value() && lineTop + height <= frame.clip->y) {
                continue;
            }
            pools.markers.push_back({
                .rect = {numbersLeft, lineTop, numbersWidth, height},
                .str = std::to_string(line - deco.first + 1),
                .visible = frame.on,
                .color = frame.marker,
                .fontSize = frame.codeGlyph,
                .alignment = {2, 1},
            });
        }
    }

    void addQuoteBar(const DecorationFrame& frame, const Deco& deco, DecorationPools& pools) const {
        const F32 top = frame.lineTop(deco.first);
        const F32 bottom = frame.lineBottom(deco.last);
        pools.boxes.push_back({
            .rect = {frame.rect.x, top, frame.body * kQuoteBarRatio, bottom - top},
            .visible = frame.on,
            .backgroundColor = frame.bar,
        });
    }

    void addListMarker(const DecorationFrame& frame, const Deco& deco, DecorationPools& pools) const {
        const F32 top = frame.lineTop(deco.first);
        const F32 height = rowHeight(deco.first);
        const F32 textStart = frame.rect.x + frame.bodyPad + deco.indent - frame.scrollX;
        if (deco.marker.empty()) {
            const F32 dot = std::max(2.0f, frame.body * kBulletDotRatio);
            pools.boxes.push_back({
                .rect = {textStart - frame.body * kBulletGapEmRatio - dot, top + height * 0.5f - dot * 0.5f, dot, dot},
                .visible = frame.on,
                .backgroundColor = frame.marker,
            });
            return;
        }
        const F32 right = textStart - frame.body * kMarkerGapEmRatio;
        const F32 width = frame.body * kMarkerWidthEmRatio;
        pools.markers.push_back({
            .rect = {right - width, top, width, height},
            .str = deco.marker,
            .visible = frame.on,
            .color = frame.marker,
            .fontSize = frame.body,
            .alignment = {2, 1},
        });
    }

    F32 tableRowBottom(const DecorationFrame& frame, const Table& table, U64 row) const {
        F32 bottom = frame.lineTop(table.cell(row, 0).line);
        for (U64 c = 0; c < table.columns; ++c) {
            bottom = std::max(bottom, frame.lineBottom(table.cell(row, c).line));
        }
        return bottom;
    }

    void addTableChrome(const DecorationFrame& frame, const Deco& deco, DecorationPools& pools) const {
        const Table& table = document.tables[deco.table];
        if (table.columns == 0 || table.rows == 0 || table.columnX.size() != table.columns + 1) {
            return;
        }
        const F32 left = frame.rect.x + frame.bodyPad - frame.scrollX;
        const F32 halfRowPad = frame.body * kTableRowPadRatio * 0.5f;
        const F32 border = std::max(1.0f, frame.body * kTableBorderRatio);
        const F32 top = frame.lineTop(table.cell(0, 0).line) - halfRowPad;
        const F32 bottom = tableRowBottom(frame, table, table.rows - 1) + halfRowPad;
        const F32 visibleTop = frame.clip.has_value() ? std::max(top, frame.clip->y) : top;
        const F32 visibleBottom = frame.clip.has_value() ? std::min(bottom, frame.clip->bottom()) : bottom;
        if (visibleBottom <= visibleTop) {
            return;
        }
        const auto push = [&](Rect rect, const ColorRGBA<F32>& color) {
            if (pools.tableBoxes.size() < kMaxDecorations) {
                pools.tableBoxes.push_back({.rect = rect, .visible = frame.on, .backgroundColor = color});
            }
        };
        const auto horizontal = [&](F32 y) {
            if (y + border * 0.5f >= visibleTop && y - border * 0.5f <= visibleBottom) {
                push({left, y - border * 0.5f, table.width, border}, frame.rule);
            }
        };

        const F32 headerBottom = table.rows > 1 ? frame.lineTop(table.cell(1, 0).line) - halfRowPad : bottom;
        if (headerBottom > visibleTop && top < visibleBottom) {
            const F32 fillTop = std::max(top, visibleTop);
            push({left, fillTop, table.width, std::min(headerBottom, visibleBottom) - fillTop}, frame.codeBackground);
        }
        horizontal(top);
        for (U64 r = 1; r < table.rows; ++r) {
            const F32 y = frame.lineTop(table.cell(r, 0).line) - halfRowPad;
            if (y > visibleBottom) {
                break;
            }
            horizontal(y);
        }
        horizontal(bottom);
        for (U64 c = 0; c <= table.columns; ++c) {
            push({left + table.columnX[c] - border * 0.5f, visibleTop, border, visibleBottom - visibleTop}, frame.rule);
        }
    }

    DecorationPools buildDecorations(const DecorationFrame& frame) const {
        DecorationPools pools;
        for (const auto& deco : document.decos) {
            if (pools.full()) {
                break;
            }
            switch (deco.kind) {
                case Deco::Kind::Rule: addRule(frame, deco, pools); break;
                case Deco::Kind::Code: addCodeBlock(frame, deco, pools); break;
                case Deco::Kind::Quote: addQuoteBar(frame, deco, pools); break;
                case Deco::Kind::ListItem: addListMarker(frame, deco, pools); break;
                case Deco::Kind::Table: addTableChrome(frame, deco, pools); break;
                case Deco::Kind::Callout: addCallout(frame, deco, pools); break;
            }
        }
        return pools;
    }

    void uploadDecorations(const DecorationFrame& frame, DecorationPools&& pools) {
        for (U64 tone = 0; tone < kCalloutTones; ++tone) {
            auto border = frame.callout[tone];
            border.a *= kCalloutBorderAlpha;
            calloutChrome[tone].update({
                .id = config.id + ":callout-" + kCalloutColorKeys[tone],
                .instances = std::move(pools.callouts[tone]),
                .clip = frame.clip,
                .cornerRadius = std::max(1.0f, frame.body * kCalloutRadiusRatio),
                .borderWidth = std::max(1.0f, std::round(frame.body * kCalloutBorderRatio * 2.0f) * 0.5f),
                .borderColor = border,
                .capacity = kMaxCallouts,
            });
        }
        codeChrome.update({
            .id = config.id + ":code-chrome",
            .instances = std::move(pools.codeBoxes),
            .clip = frame.clip,
            .cornerRadius = std::max(1.0f, frame.body * kCalloutRadiusRatio),
            .borderWidth = frame.codeBorder,
            .borderColor = frame.codeBlockBorder,
            .capacity = kMaxDecorations,
        });
        tableChrome.update({
            .id = config.id + ":table-chrome",
            .instances = std::move(pools.tableBoxes),
            .clip = frame.clip,
            .capacity = kMaxDecorations,
        });
        decorations.update({
            .id = config.id + ":deco",
            .instances = std::move(pools.boxes),
            .clip = frame.clip,
            .cornerRadius = std::max(1.0f, frame.body * kDecorationCornerRatio),
            .capacity = kMaxDecorations,
        });
        markers.update({
            .id = config.id + ":markers",
            .instances = std::move(pools.markers),
            .clip = frame.clip,
            .fontName = kCodeFont,
            .maxCharacters = kMaxMarkerCharacters,
            .capacity = kMaxDecorations,
        });
    }
};

TextMarkdown::TextMarkdown() {
    this->impl = std::make_unique<Impl>();
    setClipsChildren(true);
    for (auto& chrome : this->impl->calloutChrome) {
        add(chrome);
    }
    add(this->impl->codeChrome);
    add(this->impl->tableChrome);
    add(this->impl->decorations);
    add(this->impl->grid);
    add(this->impl->markers);
}

TextMarkdown::~TextMarkdown() = default;

bool TextMarkdown::update(Config config) {
    const bool changed = !impl->parsed ||
                         impl->config.value != config.value ||
                         impl->config.fontSize != config.fontSize ||
                         impl->config.padding != config.padding ||
                         impl->config.styleScales != config.styleScales;
    const bool widthInputsChanged = impl->config.styleFonts != config.styleFonts ||
                                    impl->config.styleScales != config.styleScales ||
                                    impl->config.styleBackgroundColorKeys != config.styleBackgroundColorKeys;
    impl->config = std::move(config);
    if (widthInputsChanged) {
        impl->tablesLayoutValid = false;
    }
    if (changed) {
        impl->rebuild();
        impl->parsed = true;
        invalidate(Dirty::Paint);
    }
    return true;
}

F32 TextMarkdown::naturalWidth() const {
    return impl->grid.naturalWidth();
}

const TextGrid::Metrics& TextMarkdown::metrics() const {
    return impl->grid.metrics();
}

Extent2D<F32> TextMarkdown::measure(const Context& ctx, Extent2D<F32> available) {
    impl->textMetrics.setWindow(ctx.render);
    impl->grid.update(impl->buildGridConfig());
    return measureChild(impl->grid, ctx, available);
}

void TextMarkdown::layout(const Context& ctx) {
    const Rect bounds = frame();
    const Rect clipPixel = Intersect(bounds, clip());

    impl->textMetrics.setWindow(ctx.render);
    impl->grid.update(impl->buildGridConfig());
    layoutChild(ctx, impl->grid, bounds);

    const auto decoFrame = impl->decorationFrame(ctx, bounds, clipPixel);
    impl->uploadDecorations(decoFrame, impl->buildDecorations(decoFrame));

    for (auto& chrome : impl->calloutChrome) {
        layoutChild(ctx, chrome, bounds);
    }
    layoutChild(ctx, impl->codeChrome, bounds);
    layoutChild(ctx, impl->tableChrome, bounds);
    layoutChild(ctx, impl->decorations, bounds);
    layoutChild(ctx, impl->markers, bounds);
}

}  // namespace Jetstream::Sakura::Retained
