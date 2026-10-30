#include <jetstream/render/sakura/components/text_markdown.hh>

#include <jetstream/render/sakura/clipboard.hh>

#include <jetstream/platform.hh>
#include <jetstream/render/sakura/components/box.hh>
#include <jetstream/render/sakura/components/label.hh>
#include <jetstream/render/sakura/components/text_grid.hh>
#include <jetstream/render/tools/icons_lucide.hh>

#include "../context.hh"
#include "../detail/helpers.hh"
#include "../detail/syntax_highlighter.hh"
#include "../detail/text_metrics.hh"
#include "../metrics.hh"
#include "tools/text.hh"

#include <algorithm>
#include <array>
#include <cctype>
#include <charconv>
#include <cmath>
#include <limits>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace Jetstream::Sakura {

using Jetstream::Text::SplitLines;
using Jetstream::Text::JoinLines;

namespace {

using StyleId = TextGrid::StyleId;
using Style = TextMarkdown::Style;

constexpr F32 kLineHeightRatio = Typography::BodyLineHeight;
constexpr F32 kParagraphGapRatio = 0.6f;
constexpr F32 kHeadingGapRatio = 1.0f;
constexpr F32 kAfterHeadingGapRatio = 0.2f;
constexpr F32 kUnderlinePadEmRatio = 0.4f;
constexpr F32 kUnderlineGapEmRatio = 0.6f;
constexpr F32 kUnderlineThicknessRatio = 1.0f / 15.0f;
constexpr F32 kListGapRatio = 0.15f;
constexpr F32 kIndentEmRatio = 1.4f;
constexpr F32 kCodePadRatio = 0.7f;
constexpr F32 kCodeInsetRatio = 1.25f;
constexpr F32 kCodeHeaderScale = 0.9f;
constexpr F32 kCodeCopyPadEmRatio = 0.6f;
constexpr F32 kCodeCopyHeightRatio = 1.8f;
constexpr const char* kCodeHeaderColorKey = "text_secondary";
constexpr F32 kDetailsInsetEmRatio = 1.0f;
constexpr F32 kDetailsPadEmRatio = 0.6f;
constexpr F32 kHeroEyebrowScale = 1.05f;
constexpr F32 kHeroTitleScale = 2.6f;
constexpr F32 kHeroSubtitleScale = 1.25f;
constexpr F32 kHeroBodyGapRatio = 1.6f;
constexpr const char* kSummaryFallback = "Details";
constexpr const char* kCodeHeaderFallback = "CODE";
constexpr const char* kCodeCopyLabel = "COPY";
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
constexpr F32 kCalloutInsetEmRatio = 1.0f;
constexpr F32 kCalloutRadiusRatio = 0.8f;
constexpr F32 kCalloutFillAlpha = 0.10f;
constexpr F32 kCalloutBorderAlpha = 0.35f;
constexpr F32 kCalloutBorderRatio = 1.5f / 15.0f;
constexpr F32 kCalloutTitleScale = 1.0f;
constexpr F32 kCalloutTitleGapEmRatio = 0.25f;
constexpr U64 kMaxCallouts = 64;
constexpr U64 kCalloutTones = Style::CalloutTones;
constexpr std::array<std::string_view, kCalloutTones> kCalloutTags = {
    "[!NOTE]", "[!TIP]", "[!IMPORTANT]", "[!WARNING]", "[!CAUTION]", "[!ERROR]",
};
constexpr std::array<const char*, kCalloutTones> kCalloutTitles = {
    "Note", "Tip", "Important", "Warning", "Caution", "Error",
};
constexpr std::array<const char*, kCalloutTones> kCalloutIcons = {
    ICON_LC_INFO, ICON_LC_LIGHTBULB, ICON_LC_MESSAGE_CIRCLE_MORE,
    ICON_LC_TRIANGLE_ALERT, ICON_LC_HAND, ICON_LC_CIRCLE_X,
};
constexpr std::array<const char*, kCalloutTones> kCalloutColorKeys = {
    "callout_note", "callout_tip", "callout_important", "callout_warning", "callout_caution", "callout_error",
};
constexpr F32 kStatPadEmRatio = 0.7f;
constexpr F32 kStatGapEmRatio = 0.6f;
constexpr F32 kStatMinTileEm = 9.0f;
constexpr F32 kStatLabelScale = 1.05f;
constexpr F32 kStatCaptionScale = 0.9f;
constexpr F32 kStatValueScale = 2.0f;
constexpr F32 kStatValueGapEmRatio = -0.2f;
constexpr F32 kStatCaptionGapEmRatio = -0.05f;
constexpr const char* kStatsTag = "stats";
constexpr const char* kStatLabelColorKey = "text_secondary";
constexpr std::array<std::pair<std::string_view, StyleId>, 12> kStatTones = {{
    {"blue", 0}, {"green", 1}, {"purple", 2}, {"yellow", 3}, {"red", 4},
    {"note", 0}, {"tip", 1}, {"important", 2}, {"warning", 3}, {"caution", 4}, {"error", 5}, {"orange", 3},
}};
constexpr F32 kFootnoteScale = 0.875f;
constexpr F32 kFootnoteRefScale = 0.75f;
constexpr F32 kFootnoteRefRaise = 0.3f;
constexpr F32 kFootnoteGapRatio = 0.3f;
constexpr const char* kFootnoteColorKey = "text_secondary";
constexpr const char* kFootnoteBackref = ICON_LC_CORNER_DOWN_LEFT;
constexpr const char* kMutedColorKey = "text_secondary";
constexpr U64 kMaxDecorations = 256;
constexpr U64 kMaxMarkerCharacters = 8;
constexpr U64 kMaxHeaderCharacters = 32;
constexpr U64 kGridLineSegments = 32;
constexpr const char* kBodyFont = TextMarkdown::BodyFont;
constexpr const char* kCodeFont = Typography::MonoFont;

constexpr std::array<std::pair<std::string_view, StyleId>, 5> kEmphasisDelimiters = {{
    {"***", Style::BoldItalic},
    {"**", Style::Bold},
    {"__", Style::Bold},
    {"*", Style::Italic},
    {"_", Style::Italic},
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
        Stats,
        Summary,
        DetailsEnd,
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
    U64 columns = 0;
    U64 level = 0;
    U64 detail = 0;
    bool open = false;
    std::string tag;
    bool underline = false;
    std::optional<SyntaxHighlighter::Language> syntax;

    bool inlineParse() const { return kind != Kind::Code; }
};

F32 HeadingGap(const Block& heading, F32 body) {
    return heading.underline ? body * (kUnderlinePadEmRatio + kUnderlineGapEmRatio)
                             : body * kLineHeightRatio * kAfterHeadingGapRatio;
}

struct LinkSpan {
    U64 start = 0;
    U64 end = 0;
    std::string url;
    std::optional<U64> line;
    U64 column = 0;
};

struct FootnoteSource {
    std::string label;
    std::vector<std::string> lines;
};

struct Footnotes {
    struct Citation {
        U64 line = 0;
        U64 column = 0;
    };

    const std::vector<FootnoteSource>* sources = nullptr;
    std::vector<U64> order;
    std::vector<std::vector<Citation>> citations;
    std::vector<std::pair<U64, U64>> pending;

    std::optional<U64> cite(std::string_view label, U64 column) {
        if (sources == nullptr) {
            return std::nullopt;
        }
        const auto source = std::find_if(sources->begin(), sources->end(), [&](const auto& entry) {
            return entry.label == label;
        });
        if (source == sources->end()) {
            return std::nullopt;
        }
        const U64 index = static_cast<U64>(source - sources->begin());
        auto number = std::find(order.begin(), order.end(), index);
        if (number == order.end()) {
            order.push_back(index);
            citations.emplace_back();
            number = std::prev(order.end());
        }
        const U64 slot = static_cast<U64>(number - order.begin());
        citations[slot].push_back({.column = column});
        pending.push_back({slot, citations[slot].size() - 1});
        return slot + 1;
    }

    void resolve(U64 line) {
        for (const auto& [slot, citation] : pending) {
            citations[slot][citation].line = line;
        }
        pending.clear();
    }
};

struct Deco {
    enum class Kind {
        Code,
        Quote,
        Rule,
        Underline,
        ListItem,
        Table,
        Callout,
        Stats,
        Details,
    };

    Kind kind = Kind::Code;
    U64 first = 0;
    U64 last = 0;
    std::string marker;
    F32 indent = 0.0f;
    U64 table = 0;
    StyleId tone = 0;
    std::string label;
    std::string text;
    F32 inset = 0.0f;
    F32 tail = 0.0f;
    bool open = false;
};

struct TableCell {
    U64 line = 0;
    std::string text;
    std::vector<StyleId> styles;
    F32 width = 0.0f;
};

struct StatTile {
    U64 first = 0;
    U64 last = 0;
    std::optional<StyleId> tone;
    F32 x = 0.0f;
};

struct StatGrid {
    U64 requested = 0;
    U64 columns = 1;
    F32 tileWidth = 0.0f;
    F32 width = 0.0f;
    F32 inset = 0.0f;
    std::vector<StatTile> tiles;
};

struct Table {
    U64 columns = 0;
    U64 rows = 0;
    std::vector<I8> aligns;
    std::vector<TableCell> cells;
    std::vector<F32> columnX;
    F32 width = 0.0f;
    F32 inset = 0.0f;

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

std::string_view TrimCellView(std::string_view s) {
    const U64 begin = s.find_first_not_of(" \t");
    if (begin == std::string_view::npos) {
        return {};
    }
    const U64 end = s.find_last_not_of(" \t");
    return s.substr(begin, end - begin + 1);
}

bool IsCommentLine(const std::string& line) {
    const std::string_view trimmed = TrimCellView(line);
    return trimmed.size() >= 7 && trimmed.starts_with("<!--") && trimmed.ends_with("-->");
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

std::optional<StyleId> ParseStatTone(const std::string& name) {
    const std::string lower = Jetstream::Text::ToLower(std::string(TrimCellView(name)));
    for (const auto& [tag, tone] : kStatTones) {
        if (lower == tag) {
            return tone;
        }
    }
    return std::nullopt;
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

std::vector<std::string> SplitCells(const std::string& line, bool& pipe) {
    std::vector<std::string> cells;
    std::string current;
    pipe = false;
    for (U64 i = 0; i < line.size(); ++i) {
        const char c = line[i];
        if (c == '\\') {
            U64 run = 1;
            while (i + run < line.size() && line[i + run] == '\\') {
                ++run;
            }
            const bool escapes = (run % 2 == 1) && i + run < line.size();
            const bool escapesPipe = escapes && line[i + run] == '|';
            current.append(escapesPipe ? run - 1 : run, '\\');
            if (escapes) {
                current += line[i + run];
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
    for (auto& cell : cells) {
        cell = TrimCell(cell);
    }
    return cells;
}

std::vector<std::string> SplitTableRow(const std::string& line) {
    bool pipe = false;
    auto cells = SplitCells(line, pipe);
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

bool ParseFootnoteDefinition(const std::string& line, std::string& label, std::string& content) {
    if (!line.starts_with("[^")) {
        return false;
    }
    const U64 close = line.find(']', 2);
    if (close == std::string::npos || close == 2 || close + 1 >= line.size() || line[close + 1] != ':') {
        return false;
    }
    label = line.substr(2, close - 2);
    content = TrimCell(line.substr(close + 2));
    return true;
}

bool IsFootnoteDefinition(const std::string& line) {
    std::string label;
    std::string content;
    return ParseFootnoteDefinition(line, label, content);
}

bool IsFootnoteContinuation(const std::string& line) {
    return line.starts_with("    ") || line.starts_with("\t");
}

bool IsEscapable(const char c) {
    return std::ispunct(static_cast<unsigned char>(c)) != 0;
}

U64 FindUnescaped(const std::string& line, std::string_view needle, const U64 from) {
    for (U64 at = line.find(needle, from); at != std::string::npos; at = line.find(needle, at + 1)) {
        U64 slashes = 0;
        while (slashes < at && line[at - 1 - slashes] == '\\') {
            ++slashes;
        }
        if (slashes % 2 == 0) {
            return at;
        }
    }
    return std::string::npos;
}

bool StartsBlock(const std::string& line) {
    return IsFence(line) || IsRule(line) || HeadingLevel(line) > 0 || IsQuoteLine(line) || IsListItem(line) ||
           IsFootnoteDefinition(line) || IsCommentLine(line);
}

std::string AutolinkLabel(const std::string& url) {
    std::string_view path = url;
    bool matched = false;
    for (const std::string_view prefix : {"https://github.com/", "http://github.com/", "https://www.github.com/"}) {
        if (path.starts_with(prefix)) {
            path.remove_prefix(prefix.size());
            matched = true;
            break;
        }
    }
    if (!matched) {
        return url;
    }
    std::vector<std::string_view> parts;
    while (!path.empty()) {
        const U64 slash = path.find('/');
        parts.push_back(path.substr(0, slash));
        path = slash == std::string_view::npos ? std::string_view{} : path.substr(slash + 1);
    }
    if (parts.size() == 4 && (parts[2] == "pull" || parts[2] == "issues") && !parts[3].empty() &&
        std::all_of(parts[3].begin(), parts[3].end(), [](char c) { return std::isdigit(static_cast<unsigned char>(c)); })) {
        return "#" + std::string(parts[3]);
    }
    if (parts.size() == 4 && parts[2] == "compare" && !parts[3].empty()) {
        return std::string(parts[3]);
    }
    return url;
}

void ParseInline(const std::string& line, StyleId baseStyle, std::string& display, std::vector<StyleId>& styles,
                 std::vector<LinkSpan>& links, Footnotes* footnotes = nullptr) {
    const U64 n = line.size();
    const auto emit = [&](char c, StyleId s) {
        display += c;
        styles.push_back(s);
    };
    const auto emitRange = [&](U64 a, U64 b, StyleId s) {
        for (U64 k = a; k < b; ++k) {
            if (s != Style::Code && line[k] == '\\' && k + 1 < b && IsEscapable(line[k + 1])) {
                ++k;
            }
            emit(line[k], s);
        }
    };
    const auto isSpace = [](char ch) {
        return ch == ' ' || ch == '\t';
    };
    const auto isWord = [](char ch) {
        return std::isalnum(static_cast<unsigned char>(ch)) != 0;
    };
    const auto emphasis = [&](U64 i, std::string_view mark, StyleId s) -> U64 {
        const U64 len = mark.size();
        const bool underscore = mark.front() == '_';
        if (line.compare(i, len, mark) != 0 || i + len >= n || isSpace(line[i + len])) {
            return 0;
        }
        if (underscore && i > 0 && isWord(line[i - 1])) {
            return 0;
        }
        for (U64 j = FindUnescaped(line, mark, i + len + 1); j != std::string::npos;
             j = FindUnescaped(line, mark, j + 1)) {
            if (isSpace(line[j - 1]) || (underscore && j + len < n && isWord(line[j + len]))) {
                continue;
            }
            const StyleId nested = baseStyle <= Style::BoldItalic ? static_cast<StyleId>(baseStyle | s) : s;
            ParseInline(line.substr(i + len, j - i - len), nested, display, styles, links, footnotes);
            return j + len;
        }
        return 0;
    };

    U64 i = 0;
    while (i < n) {
        const char c = line[i];
        if (c == '\\' && i + 1 < n && IsEscapable(line[i + 1])) {
            emit(line[i + 1], baseStyle);
            i += 2;
            continue;
        }
        if (c == '`') {
            const U64 j = line.find('`', i + 1);
            if (j != std::string::npos) {
                emitRange(i + 1, j, Style::Code);
                i = j + 1;
                continue;
            }
        }
        U64 resume = 0;
        for (const auto& [mark, style] : kEmphasisDelimiters) {
            resume = emphasis(i, mark, style);
            if (resume != 0) {
                break;
            }
        }
        if (resume != 0) {
            i = resume;
            continue;
        }
        if ((line.compare(i, 7, "http://") == 0 || line.compare(i, 8, "https://") == 0) &&
            (i == 0 || !isWord(line[i - 1]))) {
            U64 end = i;
            while (end < n && !isSpace(line[end]) && line[end] != '<' && line[end] != '>') {
                ++end;
            }
            while (end > i) {
                const char last = line[end - 1];
                if (last == ')') {
                    const auto span = std::string_view(line).substr(i, end - i);
                    if (std::count(span.begin(), span.end(), '(') >= std::count(span.begin(), span.end(), ')')) {
                        break;
                    }
                } else if (std::string_view(".,;:!?'\"").find(last) == std::string_view::npos) {
                    break;
                }
                --end;
            }
            const std::string url = line.substr(i, end - i);
            const U64 startCol = display.size();
            for (const char k : AutolinkLabel(url)) {
                emit(k, Style::Link);
            }
            links.push_back({startCol, display.size(), url});
            i = end;
            continue;
        }
        if (c == '[' && footnotes != nullptr && i + 1 < n && line[i + 1] == '^') {
            const U64 close = line.find(']', i + 2);
            const U64 column = display.size();
            const auto number = close == std::string::npos
                                    ? std::nullopt
                                    : footnotes->cite(std::string_view(line).substr(i + 2, close - i - 2), column);
            if (number.has_value()) {
                for (const char k : jst::fmt::format("[{}]", *number)) {
                    emit(k, Style::FootnoteRef);
                }
                links.push_back({column, display.size(), {}, std::nullopt});
                i = close + 1;
                continue;
            }
        }
        if (c == '[') {
            const U64 close = FindUnescaped(line, "]", i + 1);
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

std::string SoftJoin(const std::vector<std::string>& lines) {
    std::string out;
    bool open = false;
    for (const auto& line : lines) {
        const std::string_view trimmed = TrimCellView(line);
        if (trimmed.empty()) {
            if (!out.empty()) {
                out += '\n';
            }
            open = false;
            continue;
        }
        if (open) {
            out += ' ';
            out += trimmed;
        } else {
            if (!out.empty()) {
                out += '\n';
            }
            out += line;
        }
        open = true;
    }
    return out;
}

struct DetailsState {
    std::string summary;
    bool open = false;
};

struct DetailsMarker {
    U64 detail = 0;
    bool open = false;
    bool end = false;
    std::string summary;
};

struct DetailsExpansion {
    const std::vector<std::string>& source;
    const std::vector<DetailsState>& previous;
    const std::vector<bool>& forced;
    std::vector<DetailsState> states;
    std::vector<std::string> lines;
    std::vector<std::optional<DetailsMarker>> markers;

    DetailsExpansion(const std::vector<std::string>& source, const std::vector<DetailsState>& previous,
                     const std::vector<bool>& forced)
        : source(source), previous(previous), forced(forced) {
        expand(0, source.size(), true);
    }

    static bool opens(const std::string& line) {
        return Jetstream::Text::ToLower(std::string(TrimCellView(line))).starts_with("<details");
    }

    static bool closes(const std::string& line) {
        const auto lowered = Jetstream::Text::ToLower(std::string(TrimCellView(line)));
        return lowered.starts_with("</details>") || (!opens(line) && lowered.ends_with("</details>"));
    }

    static std::string closingPrefix(const std::string& line) {
        const std::string_view trimmed = TrimCellView(line);
        const auto lowered = Jetstream::Text::ToLower(std::string(trimmed));
        if (lowered.starts_with("</details>") || !lowered.ends_with("</details>")) {
            return {};
        }
        return std::string(TrimCellView(trimmed.substr(0, trimmed.size() - std::string_view("</details>").size())));
    }

    static bool single(const std::string& line) {
        return opens(line) && Jetstream::Text::ToLower(line).find("</details>") != std::string::npos;
    }

    void push(std::string line, std::optional<DetailsMarker> marker = std::nullopt) {
        lines.push_back(std::move(line));
        markers.push_back(std::move(marker));
    }

    U64 closing(U64 from, U64 to) const {
        U64 depth = 1;
        bool fenced = false;
        for (U64 line = from; line < to; ++line) {
            if (IsFence(source[line])) {
                fenced = !fenced;
                continue;
            }
            if (fenced) {
                continue;
            }
            if (opens(source[line]) && !single(source[line])) {
                ++depth;
            }
            if (closes(source[line]) && --depth == 0) {
                return line;
            }
        }
        return to;
    }

    void expand(U64 from, U64 to, bool emit) {
        bool fenced = false;
        for (U64 i = from; i < to;) {
            const bool fence = IsFence(source[i]);
            fenced = fenced != fence;
            if (fence || fenced || !opens(source[i])) {
                if (emit) {
                    push(source[i]);
                }
                ++i;
                continue;
            }
            const std::string opening = Jetstream::Text::ToLower(std::string(TrimCellView(source[i])));
            const bool initiallyOpen = opening.substr(0, opening.find('>')).find(" open") != std::string::npos;
            std::string summary = kSummaryFallback;
            U64 body = i + 1;
            std::string joined;
            const bool oneLine = single(source[i]);
            for (U64 j = i; j < to && j < i + (oneLine ? 1 : 3); ++j) {
                if (j > i && (opens(source[j]) || closes(source[j]))) {
                    break;
                }
                joined += std::string(TrimCellView(source[j])) + " ";
                const auto lowered = Jetstream::Text::ToLower(joined);
                const auto start = lowered.find("<summary>");
                const auto close = lowered.find("</summary>");
                if (start != std::string::npos && close != std::string::npos && close > start) {
                    summary = std::string(TrimCellView(std::string_view(joined).substr(start + 9, close - start - 9)));
                    body = j + 1;
                    break;
                }
            }
            std::string inlined;
            if (oneLine) {
                const std::string line(TrimCellView(source[i]));
                const auto lowered = Jetstream::Text::ToLower(line);
                const auto summaryEnd = lowered.find("</summary>");
                const U64 from = summaryEnd != std::string::npos ? summaryEnd + 10 : lowered.find('>') + 1;
                const U64 tail = lowered.rfind("</details>");
                if (tail > from) {
                    inlined = std::string(TrimCellView(std::string_view(line).substr(from, tail - from)));
                }
            }
            const U64 end = oneLine ? i : closing(body, to);
            const U64 detail = states.size();
            const bool known = detail < previous.size() && previous[detail].summary == summary;
            const bool open = detail < forced.size() ? forced[detail] : known ? previous[detail].open : initiallyOpen;
            states.push_back({.summary = summary, .open = open});
            if (emit) {
                push("", DetailsMarker{.detail = detail, .open = open, .summary = summary});
                push("");
            }
            if (emit && open && !inlined.empty()) {
                push(inlined);
            }
            expand(body, end, emit && open);
            if (emit && open && end < to && !oneLine) {
                if (const std::string prefix = closingPrefix(source[end]); !prefix.empty()) {
                    push(prefix);
                }
            }
            if (emit && open) {
                push("");
                push("", DetailsMarker{.detail = detail, .open = true, .end = true});
                push("");
            }
            i = end + 1;
        }
    }
};

struct BlockScanner {
    const std::vector<std::string>& source;
    const std::vector<std::optional<DetailsMarker>>& markers;
    F32 body;
    F32 lineHeight;
    F32 indentUnit;
    std::vector<Block> blocks;
    std::vector<FootnoteSource> footnotes;
    bool prevHeading = false;
    bool prevList = false;
    bool softBreaks = false;
    U64 index = 0;

    BlockScanner(const std::vector<std::string>& source, const std::vector<std::optional<DetailsMarker>>& markers,
                 F32 body, bool softBreaks)
        : source(source), markers(markers), body(body), lineHeight(body * kLineHeightRatio),
          indentUnit(body * kIndentEmRatio), softBreaks(softBreaks) {}

    std::vector<Block> run() {
        while (index < source.size()) {
            if (scanDetails()) {
                continue;
            }
            const std::string line = TrimRight(source[index]);
            if (line.empty() || IsCommentLine(line)) {
                ++index;
                continue;
            }
            if (scanStats(line) || scanFencedCode(line) || scanFootnote(line) || scanRule(line) ||
                scanHeading(line) || scanQuote(line) || scanTable(line) || scanListItem(line)) {
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
        if (blocks.empty() || (blocks.back().kind == Block::Kind::Summary && blocks.back().open)) {
            return 0.0f;
        }
        return prevHeading ? HeadingGap(blocks.back(), body) : lineHeight * ratio;
    }

    F32 boxPad(Block::Kind kind) const {
        switch (kind) {
            case Block::Kind::Code: return body * kCodePadRatio;
            case Block::Kind::Table: return body * kTableRowPadRatio * 0.5f;
            case Block::Kind::Callout: return body * kCalloutPadEmRatio;
            case Block::Kind::Stats: return body * kStatPadEmRatio;
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

    bool marked(U64 line) const {
        return line < markers.size() && markers[line].has_value();
    }

    bool scanDetails() {
        if (!marked(index)) {
            return false;
        }
        const DetailsMarker& marker = *markers[index];
        ++index;
        Block block;
        if (marker.end) {
            block.kind = Block::Kind::DetailsEnd;
            emit(std::move(block));
            return true;
        }
        block.kind = Block::Kind::Summary;
        block.open = marker.open;
        block.detail = marker.detail;
        block.text = marker.summary;
        block.fontSize = body;
        block.baseStyle = Style::Plain;
        block.topGap = topGapFor(block.kind, gapAbove(kParagraphGapRatio));
        emit(std::move(block));
        return true;
    }

    bool scanStats(const std::string& line) {
        if (!IsFence(line) || FenceTag(line) != kStatsTag) {
            return false;
        }
        Block block;
        block.kind = Block::Kind::Stats;
        block.fontSize = body;
        block.topGap = topGapFor(block.kind, gapAbove(kParagraphGapRatio));
        const std::string_view rest = TrimCellView(std::string_view(line).substr(
            line.find(kStatsTag) + std::string_view(kStatsTag).size()));
        I64 columns = 0;
        if (!rest.empty() && std::from_chars(rest.data(), rest.data() + rest.size(), columns).ec == std::errc() &&
            columns > 0) {
            block.columns = static_cast<U64>(columns);
        }
        ++index;
        while (index < source.size() && !IsFence(source[index])) {
            const std::string tile = TrimRight(source[index]);
            ++index;
            if (TrimCellView(tile).empty()) {
                continue;
            }
            bool pipe = false;
            block.table.rows.push_back(SplitCells(tile, pipe));
        }
        if (index < source.size()) {
            ++index;
        }
        if (!block.table.rows.empty()) {
            emit(std::move(block));
        }
        return true;
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
        block.tag = std::string(FenceTag(line));
        emit(std::move(block));
        return true;
    }

    bool scanFootnote(const std::string& line) {
        FootnoteSource footnote;
        std::string content;
        if (!ParseFootnoteDefinition(line, footnote.label, content)) {
            return false;
        }
        if (!content.empty()) {
            footnote.lines.push_back(std::move(content));
        }
        ++index;
        while (index < source.size()) {
            U64 next = index;
            while (next < source.size() && TrimRight(source[next]).empty() && !marked(next)) {
                ++next;
            }
            if (next >= source.size() || !IsFootnoteContinuation(source[next])) {
                break;
            }
            footnote.lines.push_back(TrimCell(TrimRight(source[next])));
            index = next + 1;
        }
        const bool duplicate = std::any_of(footnotes.begin(), footnotes.end(), [&](const auto& entry) {
            return entry.label == footnote.label;
        });
        if (!duplicate) {
            footnotes.push_back(std::move(footnote));
        }
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
        block.level = level;
        block.underline = level == 2;
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
        const bool callout = ParseCalloutTag(quote.front(), tone);
        const auto blank = [](const std::string& value) {
            return TrimCellView(value).empty();
        };
        auto first = quote.begin() + (callout ? 1 : 0);
        auto last = quote.end();
        while (first != last && blank(*first)) {
            ++first;
        }
        while (last != first && blank(*(last - 1))) {
            --last;
        }
        if (callout) {
            block.kind = Block::Kind::Callout;
            block.tone = tone;
            block.text = softBreaks ? SoftJoin({first, last}) : JoinLines({first, last});
            block.indent = body * kCalloutInsetEmRatio;
        } else {
            block.kind = Block::Kind::Quote;
            block.text = first == last ? std::string() : JoinLines({first, last});
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
        while (softBreaks && index + 1 < source.size()) {
            const std::string next = TrimRight(source[index + 1]);
            const std::string_view content = TrimCellView(next);
            std::string ignored;
            if (content.empty() || StartsBlock(std::string(content)) || ListMarker(next, ignored) != 0 ||
                tableStartsAt(index + 1)) {
                break;
            }
            block.text += " " + std::string(content);
            ++index;
        }
        block.fontSize = body;
        block.topGap = topGapFor(block.kind, prevList      ? lineHeight * kListGapRatio
                                             : prevHeading ? HeadingGap(blocks.back(), body)
                                                           : lineHeight * kParagraphGapRatio);
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
        block.text = softBreaks ? SoftJoin(paragraph) : JoinLines(paragraph);
        block.fontSize = body;
        block.topGap = topGapFor(block.kind, gapAbove(kParagraphGapRatio));
        emit(std::move(block));
    }
};

struct Document {
    std::vector<F32> lineScale;
    std::vector<F32> lineTopGap;
    std::vector<F32> lineIndent;
    std::vector<TextGrid::RowMode> lineRowMode;
    std::vector<F32> lineWrapWidth;
    std::vector<F32> lineRightInset;
    std::vector<std::vector<StyleId>> styles;
    std::vector<std::vector<LinkSpan>> links;
    std::vector<Deco> decos;
    std::vector<Table> tables;
    std::vector<StatGrid> stats;
    std::vector<TextGrid::ScrollRegion> scrollRegions;
    std::vector<TextMarkdown::Heading> headings;
    std::vector<std::pair<U64, U64>> summaries;
    std::string plainValue;
    F32 trailingPad = 0.0f;

    F32 scaleAt(U64 line) const {
        return line < lineScale.size() ? lineScale[line] : 1.0f;
    }

    const LinkSpan* linkAt(TextGrid::Position pos) const {
        if (pos.line >= links.size()) {
            return nullptr;
        }
        for (const auto& span : links[pos.line]) {
            if (pos.column >= span.start && pos.column < span.end) {
                return &span;
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
    Footnotes footnotes;
    std::vector<std::string> lines;
    F32 pendingGap = 0.0f;
    F32 regionInset = 0.0f;
    std::vector<std::pair<U64, U64>> details;
    const Block* lastBlock = nullptr;
    F32 closedTail = 0.0f;
    std::optional<U64> closedAt;

    DocumentBuilder(F32 body, F32 codeIndent, F32 codeRightInset, SyntaxHighlighter& highlighter)
        : body(body), codeIndent(codeIndent), codeRightInset(codeRightInset), highlighter(highlighter) {}

    Document build(const std::vector<Block>& blocks,
                   const std::vector<FootnoteSource>& sources,
                   const std::optional<TextMarkdown::Hero>& hero) {
        footnotes.sources = &sources;
        if (hero.has_value()) {
            emitHero(*hero);
        }
        for (const auto& block : blocks) {
            if (block.kind != Block::Kind::DetailsEnd) {
                lastBlock = &block;
            }
            switch (block.kind) {
                case Block::Kind::Rule: emitRule(block); break;
                case Block::Kind::Table: emitTable(block); break;
                case Block::Kind::Callout: emitCallout(block); break;
                case Block::Kind::Stats: emitStats(block); break;
                case Block::Kind::Summary: emitSummary(block); break;
                case Block::Kind::DetailsEnd: closeDetails(); break;
                default: emitTextBlock(block); break;
            }
        }
        while (!details.empty()) {
            closeDetails();
        }
        emitFootnotes(blocks);
        if (lines.empty()) {
            addLine("", 1.0f, 0.0f, 0.0f, {}, {});
        }
        if (footnotes.order.empty()) {
            if (!blocks.empty()) {
                document.trailingPad = overhangFor(blocks.back());
            } else if (hero.has_value()) {
                document.trailingPad = underlineOverhang();
            }
        }
        document.plainValue = JoinLines(lines);
        return std::move(document);
    }

    F32 trailingPadFor(Block::Kind kind) const {
        switch (kind) {
            case Block::Kind::Code: return body * kCodePadRatio;
            case Block::Kind::Table: return body * kTableRowPadRatio * 0.5f;
            case Block::Kind::Callout: return body * kCalloutPadEmRatio;
            case Block::Kind::Stats: return body * kStatPadEmRatio;
            default: return 0.0f;
        }
    }

    F32 underlineOverhang() const {
        return body * kUnderlinePadEmRatio + std::max(1.0f, std::round(body * kUnderlineThicknessRatio));
    }

    F32 overhangFor(const Block& block) const {
        switch (block.kind) {
            case Block::Kind::Heading: return block.underline ? underlineOverhang() : 0.0f;
            case Block::Kind::Summary: return block.open ? 0.0f : body * kDetailsPadEmRatio;
            case Block::Kind::DetailsEnd: return closedTail;
            default: return trailingPadFor(block.kind);
        }
    }

    void emitFootnotes(const std::vector<Block>& blocks) {
        const F32 lineHeight = body * kLineHeightRatio;
        const F32 indent = body * kIndentEmRatio;
        std::vector<U64> itemLines;
        for (U64 slot = 0; slot < footnotes.order.size(); ++slot) {
            const auto& source = (*footnotes.sources)[footnotes.order[slot]];
            const U64 first = lines.size();
            F32 gap = lineHeight * kFootnoteGapRatio;
            if (slot == 0) {
                gap = blocks.empty() ? 0.0f
                                     : (blocks.back().kind == Block::Kind::Heading ? HeadingGap(blocks.back(), body)
                                                                                  : lineHeight * kParagraphGapRatio) +
                                           trailingPadFor(blocks.back().kind);
            }
            const auto sources = source.lines.empty() ? std::vector<std::string>{""} : source.lines;
            for (U64 li = 0; li < sources.size(); ++li) {
                std::vector<StyleId> styles;
                std::vector<LinkSpan> links;
                std::string display;
                ParseInline(sources[li], Style::Plain, display, styles, links, &footnotes);
                for (auto& style : styles) {
                    if (style <= Style::BoldItalic) {
                        style = Style::Footnote(style);
                    }
                }
                if (li + 1 == sources.size()) {
                    appendBackrefs(slot, display, styles, links);
                }
                addLine(std::move(display), kFootnoteScale, li == 0 ? gap : 0.0f, indent, std::move(styles),
                        std::move(links));
            }
            document.decos.push_back({.kind = Deco::Kind::ListItem, .first = first, .last = first,
                                      .marker = jst::fmt::format("{}.", slot + 1), .indent = indent});
            itemLines.push_back(first);
        }
        for (U64 slot = 0; slot < footnotes.citations.size() && slot < itemLines.size(); ++slot) {
            for (const auto& citation : footnotes.citations[slot]) {
                for (auto& span : document.links[citation.line]) {
                    if (span.start == citation.column && span.url.empty() && !span.line.has_value()) {
                        span.line = itemLines[slot];
                    }
                }
            }
        }
    }

    void appendBackrefs(U64 slot, std::string& display, std::vector<StyleId>& styles, std::vector<LinkSpan>& links) {
        const auto citations = footnotes.citations[slot];
        for (U64 k = 0; k < citations.size(); ++k) {
            display += ' ';
            styles.push_back(Style::Footnote(Style::Plain));
            const U64 start = display.size();
            for (const char c : std::string_view(kFootnoteBackref)) {
                display += c;
                styles.push_back(Style::Backref);
            }
            if (k > 0) {
                for (const char c : std::to_string(k + 1)) {
                    display += c;
                    styles.push_back(Style::FootnoteRef);
                }
            }
            links.push_back({start, display.size(), {}, citations[k].line, citations[k].column});
        }
    }

    void emitStats(const Block& block) {
        StatGrid grid;
        grid.requested = block.columns;
        grid.inset = regionInset;
        const F32 rowGap = body * (2.0f * kStatPadEmRatio + kStatGapEmRatio);
        const auto addStatLine = [&](const std::string& source, StyleId baseStyle, std::optional<StyleId> tone,
                                     F32 scale, F32 gap, TextGrid::RowMode mode) {
            std::vector<StyleId> styles;
            std::vector<LinkSpan> links;
            std::string display;
            ParseInline(source, baseStyle, display, styles, links, &footnotes);
            if (tone.has_value()) {
                TintStyles(styles, tone.value());
            }
            addLine(std::move(display), scale, gap, 0.0f, std::move(styles), std::move(links), mode);
        };

        for (U64 i = 0; i < block.table.rows.size(); ++i) {
            const auto& fields = block.table.rows[i];
            const bool labeled = fields.size() > 1;
            StatTile tile;
            tile.first = lines.size();
            tile.tone = fields.size() > 2 ? ParseStatTone(fields[2]) : std::nullopt;
            const F32 gap = i == 0 ? block.topGap : rowGap;
            const auto mode = i == 0 ? TextGrid::RowMode::Next : TextGrid::RowMode::Same;
            addStatLine(labeled ? fields[0] : "", tile.tone.has_value() ? Style::Bold : Style::StatLabel, tile.tone,
                        kStatLabelScale, gap, mode);
            addStatLine(labeled ? fields[1] : fields[0], Style::Bold, tile.tone, kStatValueScale,
                        body * kStatValueGapEmRatio, TextGrid::RowMode::Below);
            if (fields.size() > 3 && !TrimCellView(fields[3]).empty()) {
                addStatLine(fields[3], Style::StatLabel, std::nullopt, kStatCaptionScale, body * kStatCaptionGapEmRatio,
                            TextGrid::RowMode::Below);
            }
            tile.last = lines.size() - 1;
            grid.tiles.push_back(tile);
        }
        document.decos.push_back({.kind = Deco::Kind::Stats, .table = document.stats.size()});
        document.stats.push_back(std::move(grid));
    }

    void emitHero(const TextMarkdown::Hero& hero) {
        const auto plain = [&](std::string text, StyleId style, F32 scale, F32 gap) {
            std::vector<StyleId> styles(text.size(), style);
            addLine(std::move(text), scale, gap, 0.0f, std::move(styles), {});
        };
        if (!hero.eyebrow.empty()) {
            plain(Jetstream::Text::ToUpper(hero.eyebrow), Style::Muted, kHeroEyebrowScale, 0.0f);
        }
        plain(hero.title, Style::Bold, kHeroTitleScale, 0.0f);
        if (!hero.subtitle.empty()) {
            plain(hero.subtitle, Style::Muted, kHeroSubtitleScale, 0.0f);
        }
        document.decos.push_back({.kind = Deco::Kind::Underline, .first = lines.size() - 1, .last = lines.size() - 1});
        pendingGap = body * kHeroBodyGapRatio;
    }

    void emitSummary(const Block& block) {
        std::vector<StyleId> styles;
        std::vector<LinkSpan> links;
        std::string display;
        const std::string source = jst::fmt::format("{}  {}", block.open ? ICON_LC_CHEVRON_DOWN : ICON_LC_CHEVRON_RIGHT,
                                                    block.text);
        ParseInline(source, block.baseStyle, display, styles, links, &footnotes);
        const F32 pad = body * kDetailsPadEmRatio;
        const F32 inset = body * kDetailsInsetEmRatio;
        const U64 line = lines.size();
        document.summaries.emplace_back(line, block.detail);
        addLine(std::move(display), 1.0f, block.topGap + pad, block.indent + inset, std::move(styles), std::move(links),
                TextGrid::RowMode::Next, inset);
        if (!block.open) {
            document.decos.push_back({.kind = Deco::Kind::Details, .first = line, .last = line});
            pendingGap = pad;
            return;
        }
        details.emplace_back(line, document.decos.size());
        regionInset += inset;
        pendingGap = pad + inset;
    }

    void closeDetails() {
        if (details.empty()) {
            return;
        }
        const auto [line, first] = details.back();
        details.pop_back();
        for (U64 index = first; index < document.decos.size(); ++index) {
            document.decos[index].inset = std::max(document.decos[index].inset, regionInset);
        }
        const F32 inset = body * kDetailsInsetEmRatio;
        const bool stacked = closedAt == lines.size();
        const F32 tail = (stacked ? closedTail : lastBlock != nullptr ? overhangFor(*lastBlock) : 0.0f) + inset;
        closedTail = tail;
        closedAt = lines.size();
        regionInset = std::max(0.0f, regionInset - inset);
        document.decos.push_back({.kind = Deco::Kind::Details, .first = line, .last = lines.size() - 1,
                                  .inset = regionInset, .tail = tail, .open = true});
        pendingGap = tail;
    }

    void emitCallout(const Block& block) {
        const U64 firstLine = lines.size();
        const auto addCalloutLine = [&](const std::string& source, StyleId baseStyle, F32 scale, F32 gap,
                                        bool tint) {
            std::vector<StyleId> styles;
            std::vector<LinkSpan> links;
            std::string display;
            ParseInline(source, baseStyle, display, styles, links, &footnotes);
            if (tint) {
                TintStyles(styles, block.tone);
            }
            addLine(std::move(display), scale, gap, block.indent, std::move(styles), std::move(links),
                    TextGrid::RowMode::Next, block.indent);
        };
        addCalloutLine(jst::fmt::format("{}  {}", kCalloutIcons[block.tone], kCalloutTitles[block.tone]), Style::Bold,
                       kCalloutTitleScale, block.topGap, true);
        if (!block.text.empty()) {
            F32 gap = body * kCalloutTitleGapEmRatio;
            for (const auto& source : SplitLines(block.text)) {
                if (TrimCellView(source).empty()) {
                    gap = body * kLineHeightRatio * kParagraphGapRatio;
                    continue;
                }
                addCalloutLine(source, Style::Plain, 1.0f, gap, true);
                gap = 0.0f;
            }
        }
        document.decos.push_back({.kind = Deco::Kind::Callout, .first = firstLine, .last = lines.size() - 1,
                                  .tone = block.tone});
    }

    void addLine(std::string text, F32 scale, F32 gap, F32 indent,
                 std::vector<StyleId> lineStyles, std::vector<LinkSpan> lineLinks,
                 TextGrid::RowMode rowMode = TextGrid::RowMode::Next,
                 F32 rightInset = 0.0f) {
        footnotes.resolve(lines.size());
        lines.push_back(std::move(text));
        document.lineScale.push_back(scale);
        document.lineTopGap.push_back(gap + std::exchange(pendingGap, 0.0f));
        document.lineIndent.push_back(indent + regionInset);
        document.lineRowMode.push_back(rowMode);
        document.lineWrapWidth.push_back(0.0f);
        document.lineRightInset.push_back(rightInset + regionInset);
        document.styles.push_back(std::move(lineStyles));
        document.links.push_back(std::move(lineLinks));
    }

    void emitTable(const Block& block) {
        const F32 rowPad = body * kTableRowPadRatio;
        Table table;
        table.columns = block.table.aligns.size();
        table.rows = block.table.rows.size();
        table.aligns = block.table.aligns;
        table.inset = regionInset;
        for (U64 r = 0; r < table.rows; ++r) {
            for (U64 c = 0; c < table.columns; ++c) {
                TableCell cell;
                std::vector<StyleId> styles;
                std::vector<LinkSpan> links;
                ParseInline(block.table.rows[r][c], r == 0 ? Style::Bold : Style::Plain, cell.text, styles, links,
                            &footnotes);
                cell.styles = styles;
                cell.line = lines.size();
                const F32 gap = c > 0 ? 0.0f : r == 0 ? block.topGap + rowPad * 0.5f : rowPad;
                addLine(cell.text, 1.0f, gap, 0.0f, std::move(styles), std::move(links),
                        c > 0 ? TextGrid::RowMode::Same : TextGrid::RowMode::Next);
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
        const bool code = block.kind == Block::Kind::Code;
        if (code) {
            addLine("", 1.0f, block.topGap, indent, {}, {}, TextGrid::RowMode::Next, codeRightInset);
        }
        for (U64 li = 0; li < sources.size(); ++li) {
            const F32 gap = li == 0 ? (code ? 2.0f * body * kCodePadRatio : block.topGap) : 0.0f;
            std::vector<StyleId> styles;
            std::vector<LinkSpan> links;
            std::string display;
            if (block.inlineParse()) {
                ParseInline(sources[li], block.baseStyle, display, styles, links, &footnotes);
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
            addLine(std::move(display), scale, gap, indent, std::move(styles), std::move(links),
                    TextGrid::RowMode::Next, block.kind == Block::Kind::Code ? codeRightInset : 0.0f);
        }
        const U64 lastLine = lines.size() - 1;

        switch (block.kind) {
            case Block::Kind::Code: {
                std::string label = Jetstream::Text::ToUpper(block.tag.empty() ? kCodeHeaderFallback : block.tag);
                if (label.size() > kMaxHeaderCharacters) {
                    label = std::string(Jetstream::Text::Utf8Prefix(label, kMaxHeaderCharacters - 3)) + "...";
                }
                document.decos.push_back({.kind = Deco::Kind::Code, .first = firstLine, .last = lastLine,
                                          .indent = indent, .label = std::move(label), .text = block.text});
                if (lastLine > firstLine) {
                    document.scrollRegions.push_back({.first = firstLine + 1, .last = lastLine});
                }
                break;
            }
            case Block::Kind::Quote:
                document.decos.push_back({.kind = Deco::Kind::Quote, .first = firstLine, .last = lastLine, .indent = indent});
                break;
            case Block::Kind::ListItem:
                document.decos.push_back({.kind = Deco::Kind::ListItem, .first = firstLine, .last = firstLine,
                                          .marker = block.marker, .indent = indent});
                break;
            case Block::Kind::Heading:
                document.headings.push_back({.text = lines[firstLine], .level = block.level, .line = firstLine});
                if (block.underline) {
                    document.decos.push_back({.kind = Deco::Kind::Underline, .first = lastLine, .last = lastLine});
                }
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
        F32 bodyPadRight = 0.0f;
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
        ColorRGBA<F32> label;
        ColorRGBA<F32> labelHovered;
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

    struct HitTarget {
        Rect rect;
        U64 line = 0;
        bool copy = false;

        bool operator==(const HitTarget&) const = default;
    };

    struct DecorationPools {
        std::vector<Box::Instance> boxes;
        std::vector<Box::Instance> tableBoxes;
        std::vector<Box::Instance> codeBoxes;
        std::array<std::vector<Box::Instance>, kCalloutTones> callouts;
        std::vector<Label::Instance> markers;
        std::vector<Label::Instance> headers;
        std::vector<HitTarget> targets;

        bool full() const {
            return boxes.size() >= kMaxDecorations || tableBoxes.size() >= kMaxDecorations ||
                   codeBoxes.size() >= kMaxDecorations || markers.size() >= kMaxDecorations ||
                   headers.size() >= kMaxDecorations;
        }
    };

    Config config;
    TextGrid grid;
    std::array<Box, kCalloutTones> calloutChrome;
    Box codeChrome;
    Box tableChrome;
    Box decorations;
    Label markers;
    Label headers;
    SyntaxHighlighter highlighter;
    Document document;
    mutable TextMetrics textMetrics;
    U64 styleRevision = 0;
    bool parsed = false;
    std::vector<DetailsState> details;
    std::vector<HitTarget> targets;
    std::optional<HitTarget> pressed;
    std::optional<U64> hoveredCopy;
    std::optional<Extent2D<F32>> pointer;
    TextMarkdown* owner = nullptr;
    bool tablesLayoutValid = false;
    F32 tablesLayoutWidth = -1.0f;
    std::vector<U8> tablesLayoutFonts;
    TextGrid::WidthLayout tablesLayout;
    bool gridConfigDirty = true;
    const Render::Window* gridConfigWindow = nullptr;
    U64 gridConfigFontRevision = 0;

    void rebuild() {
        ++styleRevision;
        const std::vector<std::string> source = SplitLines(config.value);
        DetailsExpansion expansion(source, details, config.detailsOpen);
        details = std::move(expansion.states);
        BlockScanner scanner(expansion.lines, expansion.markers, config.fontSize, config.softBreaks);
        const std::vector<Block> blocks = scanner.run();
        const F32 inset = codeInset();
        document = DocumentBuilder(config.fontSize, inset, inset, highlighter)
                       .build(blocks, scanner.footnotes, config.hero);
        tablesLayoutValid = false;
        gridConfigDirty = true;
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
        return document.lineIndent;
    }

    F32 minimumContentWidth(const std::vector<F32>& indents) const {
        const F32 glyph = glyphMinimumWidth();
        F32 widest = widestTableWidth();
        for (const auto& grid : document.stats) {
            widest = std::max(widest, grid.width + 2.0f * grid.inset);
        }
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

    std::optional<U64> summaryAt(U64 line) const {
        for (const auto& [summaryLine, detail] : document.summaries) {
            if (summaryLine == line && detail < details.size()) {
                return detail;
            }
        }
        return std::nullopt;
    }

    const Deco* codeHeaderAt(U64 line) const {
        for (const auto& deco : document.decos) {
            if (deco.kind == Deco::Kind::Code && deco.first == line) {
                return &deco;
            }
        }
        return nullptr;
    }

    F32 codeInset() const {
        return config.fontSize * kCodeInsetRatio;
    }

    static StyleId configStyle(StyleId id) {
        if (id >= Style::SyntaxBase && id < Style::SyntaxBase + Style::SyntaxStyles) {
            return Style::CodeBlock;
        }
        if (id >= Style::FootnoteBase && id < Style::FootnoteRef) {
            return static_cast<StyleId>(id - Style::FootnoteBase);
        }
        return id;
    }

    F32 styleScale(StyleId style) const {
        if (style == Style::FootnoteRef) {
            return kFootnoteRefScale;
        }
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
            if (!font.empty() && !textMetrics.loaded(font)) {
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
            widest = std::max(widest, table.width + 2.0f * table.inset);
        }
        return widest;
    }

    TextGrid::WidthLayout layoutForTextWidth(F32 textWidth) {
        auto fonts = FontAvailability(textMetrics.window, kBodyFont, config.styleFonts);
        if (!(tablesLayoutValid && tablesLayoutWidth == textWidth && tablesLayoutFonts == fonts)) {
            layoutTables(textWidth);
            layoutStats(textWidth);
            std::vector<F32> indents = lineIndents();
            const F32 minimumWidth = minimumContentWidth(indents);
            tablesLayout = {
                .lineIndent = std::move(indents),
                .lineWrapWidth = document.lineWrapWidth,
                .lineRowMode = document.lineRowMode,
                .contentMinWidth = minimumWidth,
            };
            tablesLayoutValid = fontsReady();
            tablesLayoutWidth = textWidth;
            tablesLayoutFonts = std::move(fonts);
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
        const F32 glyphMinimum = glyphMinimumWidth();

        for (auto& table : document.tables) {
            if (table.columns == 0 || table.rows == 0) {
                continue;
            }
            const F32 available = bounded ? std::max(1.0f, textWidth - 2.0f * table.inset)
                                          : std::numeric_limits<F32>::max();
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
                document.lineIndent[cell.line] = table.inset + table.columnX[c] + padX + shift;
                document.lineWrapWidth[cell.line] = columnWidth + kTableWrapSlack;
            }
        }
    }

    void layoutStats(F32 textWidth) {
        const F32 body = config.fontSize;
        const F32 gap = body * kStatGapEmRatio;
        const F32 pad = body * kStatPadEmRatio;
        const F32 minTile = body * kStatMinTileEm;
        for (auto& grid : document.stats) {
            const F32 width = std::isfinite(textWidth) ? textWidth - 2.0f * grid.inset : textWidth;
            const bool bounded = std::isfinite(width) && width > 0.0f;
            const U64 count = grid.tiles.size();
            U64 columns = grid.requested;
            if (columns == 0) {
                columns = bounded ? static_cast<U64>(std::max(1.0f, std::floor((width + gap) / (minTile + gap))))
                                  : count;
            }
            grid.columns = std::clamp<U64>(columns, 1, count);
            const F32 gaps = gap * static_cast<F32>(grid.columns - 1);
            grid.tileWidth = bounded ? std::max(2.0f * pad + glyphMinimumWidth(),
                                                (width - gaps) / static_cast<F32>(grid.columns))
                                     : minTile;
            grid.width = grid.tileWidth * static_cast<F32>(grid.columns) + gaps;

            for (U64 i = 0; i < count; ++i) {
                auto& tile = grid.tiles[i];
                const U64 column = i % grid.columns;
                tile.x = static_cast<F32>(column) * (grid.tileWidth + gap);
                for (U64 line = tile.first; line <= tile.last; ++line) {
                    document.lineIndent[line] = grid.inset + tile.x + pad;
                    document.lineWrapWidth[line] = std::max(1.0f, grid.tileWidth - 2.0f * pad);
                    if (line == tile.first) {
                        document.lineRowMode[line] = i == 0 || column == 0 ? TextGrid::RowMode::Next
                                                                           : TextGrid::RowMode::Same;
                    }
                }
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
        keys.emplace_back(kStatLabelColorKey);
        keys.resize(Style::FootnoteRef - 1, kFootnoteColorKey);
        const std::string link =
            config.styleColorKeys.size() >= Style::Link ? config.styleColorKeys[Style::Link - 1] : "";
        keys.push_back(link);
        keys.emplace_back(kMutedColorKey);
        keys.push_back(link);
        return keys;
    }

    std::vector<F32> gridStyleScales() const {
        auto scales = config.styleScales;
        scales.resize(Style::SyntaxBase - 1, 0.0f);
        scales.resize(Style::SyntaxBase - 1 + Style::SyntaxStyles, styleScale(Style::CodeBlock));
        scales.push_back(0.0f);
        scales.resize(Style::FootnoteRef - 1, 0.0f);
        scales.push_back(kFootnoteRefScale);
        scales.push_back(0.0f);
        scales.push_back(0.0f);
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
        fonts.push_back(styleFont(Style::Bold));
        for (StyleId emphasis = Style::Plain; emphasis <= Style::BoldItalic; ++emphasis) {
            fonts.push_back(emphasis == Style::Plain ? "" : styleFont(emphasis));
        }
        fonts.emplace_back("");
        fonts.emplace_back("");
        fonts.emplace_back("");
        return fonts;
    }

    static std::vector<bool> gridStyleUnderlines() {
        std::vector<bool> underlines(Style::Link, false);
        underlines.back() = true;
        return underlines;
    }

    static std::vector<F32> gridStyleRaises() {
        std::vector<F32> raises(Style::FootnoteRef, 0.0f);
        raises.back() = kFootnoteRefRaise;
        return raises;
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
            .scrollbarInset = config.scrollbarInset,
            .wrap = TextGrid::Wrap::Word,
            .padding = gridPadding(),
            .lineScale = document.lineScale,
            .lineTopGap = document.lineTopGap,
            .lineIndent = std::move(indents),
            .lineRowMode = document.lineRowMode,
            .lineWrapWidth = document.lineWrapWidth,
            .lineRightInset = document.lineRightInset,
            .contentMinWidth = minimumWidth,
            .scrollRegions = document.scrollRegions,
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
            .styleRaises = gridStyleRaises(),
            .styleUnderlines = gridStyleUnderlines(),
            .maxLineSegments = kGridLineSegments,
            .styleRevision = styleRevision,
            .styler = [this](const std::vector<std::string>&, U64)
                          -> const std::vector<std::vector<StyleId>>& {
                return document.styles;
            },
            .widthLayout = document.tables.empty() && document.stats.empty()
                ? std::function<TextGrid::WidthLayout(F32)>{}
                : [this](F32 textWidth) { return layoutForTextWidth(textWidth); },
            .onPositionClick = [this](TextGrid::Position pos) -> bool {
                const LinkSpan* span = document.linkAt(pos);
                if (span == nullptr) {
                    return false;
                }
                if (span->line.has_value()) {
                    const TextGrid::Position target = {*span->line, span->column};
                    if (!(config.onReveal && config.onReveal(grid.positionTop(target)))) {
                        grid.scrollToLine(target.line, target.column);
                    }
                } else if (!span->url.empty() && !(config.onLink && config.onLink(span->url))) {
                    (void)Platform::OpenUrl(span->url);
                }
                return true;
            },
            .isPositionInteractive = [this](TextGrid::Position pos) -> bool {
                return document.linkAt(pos) != nullptr;
            },
        };
    }

    void syncGrid() {
        const auto* window = textMetrics.window;
        const U64 fonts = window ? window->fontRevision() : 0;
        if (!gridConfigDirty && gridConfigWindow == window && gridConfigFontRevision == fonts) {
            return;
        }
        grid.update(buildGridConfig());
        gridConfigDirty = false;
        gridConfigWindow = window;
        gridConfigFontRevision = fonts;
    }

    DecorationFrame decorationFrame(const Context& ctx, const Rect& rect, const Rect& clipPixel) const {
        const auto& metrics = grid.metrics();
        const F32 body = config.fontSize;
        return {
            .rect = rect,
            .clip = clipPixel,
            .body = body,
            .bodyPad = metrics.padding.left,
            .bodyPadRight = metrics.padding.right,
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
            .label = ctx.color(kCodeHeaderColorKey),
            .labelHovered = ctx.color(config.textColorKey),
            .callout = calloutColors(ctx),
            .scrollX = metrics.scrollX + metrics.overscroll.x,
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
        fill = WithAlpha(fill, kCalloutFillAlpha);
        instances.push_back({
            .rect = {frame.rect.x + frame.bodyPad, top,
                     std::max(0.0f, frame.width - frame.bodyPad - frame.bodyPadRight), bottom - top},
            .visible = frame.on,
            .backgroundColor = fill,
        });
    }

    void addRule(const DecorationFrame& frame, const Deco& deco, DecorationPools& pools) const {
        const F32 thickness = std::max(1.0f, frame.body * kRuleThicknessRatio);
        const F32 center = frame.lineTop(deco.first) + rowHeight(deco.first) * 0.5f;
        pools.boxes.push_back({
            .rect = {frame.rect.x + frame.bodyPad, center - thickness * 0.5f,
                     std::max(0.0f, frame.width - frame.bodyPad - frame.bodyPadRight), thickness},
            .visible = frame.on,
            .backgroundColor = frame.rule,
        });
    }

    void addUnderline(const DecorationFrame& frame, const Deco& deco, DecorationPools& pools) const {
        const F32 thickness = std::max(1.0f, std::round(frame.body * kUnderlineThicknessRatio));
        const F32 y = std::round(frame.lineBottom(deco.last) + frame.body * kUnderlinePadEmRatio);
        pools.boxes.push_back({
            .rect = {frame.rect.x + frame.bodyPad, y, std::max(0.0f, frame.width - frame.bodyPad - frame.bodyPadRight),
                     thickness},
            .visible = frame.on,
            .backgroundColor = frame.rule,
        });
    }

    void addCodeBlock(const DecorationFrame& frame, const Deco& deco, DecorationPools& pools) const {
        const F32 pad = frame.body * kCodePadRatio;
        const F32 top = frame.lineTop(deco.first);
        const F32 bottom = frame.lineBottom(deco.last);
        const F32 left = frame.rect.x + frame.bodyPad;
        const F32 width = std::max(0.0f, frame.width - frame.bodyPad - frame.bodyPadRight);
        pools.codeBoxes.push_back({
            .rect = {left, top - pad, width, (bottom - top) + 2.0f * pad},
            .visible = frame.on,
            .backgroundColor = frame.codeBlock,
        });

        const F32 headerTop = top - pad;
        const F32 headerBottom = frame.lineBottom(deco.first) + pad;
        if (deco.last > deco.first) {
            pools.boxes.push_back({
                .rect = {left + frame.codeBorder, std::round(headerBottom - frame.codeBorder * 0.5f),
                         std::max(0.0f, width - 2.0f * frame.codeBorder), frame.codeBorder},
                .visible = frame.on,
                .backgroundColor = frame.codeBlockBorder,
            });
        }
        if (frame.clip.has_value() && (headerTop >= frame.clip->bottom() || headerBottom <= frame.clip->y)) {
            return;
        }
        const F32 inset = deco.indent;
        const Rect header = {left + inset, headerTop, std::max(0.0f, width - 2.0f * inset), headerBottom - headerTop};
        const F32 size = frame.codeGlyph * kCodeHeaderScale;
        const F32 copyPad = size * kCodeCopyPadEmRatio;
        const F32 copyWidth = textMetrics.measure(kCodeFont, kCodeCopyLabel, size) + 2.0f * copyPad;
        const F32 copyHeight = std::min(size * kCodeCopyHeightRatio, header.height);
        const F32 right = left + width - pad;
        const Rect copy = {std::round(right - copyWidth), std::round(header.y + (header.height - copyHeight) * 0.5f),
                           std::round(copyWidth), std::round(copyHeight)};
        const F32 labelWidth = std::max(0.0f, copy.x - copyPad - header.x);
        pools.headers.push_back({
            .rect = {header.x, header.y, labelWidth, header.height},
            .str = textMetrics.ellipsize(kCodeFont, deco.label, size, labelWidth),
            .visible = frame.on,
            .color = frame.label,
            .fontSize = size,
            .alignment = {0, 1},
        });
        const bool hovered = hoveredCopy == deco.first;
        pools.codeBoxes.push_back({
            .rect = copy,
            .visible = frame.on,
            .backgroundColor = hovered ? frame.codeBlockBorder : frame.codeBlock,
        });
        pools.headers.push_back({
            .rect = copy,
            .str = kCodeCopyLabel,
            .visible = frame.on,
            .color = hovered ? frame.labelHovered : frame.label,
            .fontSize = size,
            .alignment = {1, 1},
        });
        if (frame.on) {
            pools.targets.push_back({.rect = copy, .line = deco.first, .copy = true});
        }
    }

    void addDetails(const DecorationFrame& frame, const Deco& deco, DecorationPools& pools) const {
        const F32 pad = frame.body * kDetailsPadEmRatio;
        const F32 top = frame.lineTop(deco.first) - pad;
        const F32 bottom = frame.lineBottom(deco.last) + (deco.open ? deco.tail : pad);
        const F32 left = frame.rect.x + frame.bodyPad;
        const F32 width = std::max(0.0f, frame.width - frame.bodyPad - frame.bodyPadRight);
        pools.codeBoxes.push_back({
            .rect = {left, top, width, bottom - top},
            .visible = frame.on,
            .backgroundColor = {0.0f, 0.0f, 0.0f, 0.0f},
        });
        if (frame.on) {
            pools.targets.push_back({.rect = {left, top, width, frame.lineBottom(deco.first) + pad - top},
                                     .line = deco.first});
        }
        if (deco.open) {
            pools.boxes.push_back({
                .rect = {left + frame.codeBorder, std::round(frame.lineBottom(deco.first) + pad - frame.codeBorder * 0.5f),
                         std::max(0.0f, width - 2.0f * frame.codeBorder), frame.codeBorder},
                .visible = frame.on,
                .backgroundColor = frame.codeBlockBorder,
            });
        }
    }

    void addQuoteBar(const DecorationFrame& frame, const Deco& deco, DecorationPools& pools) const {
        const F32 top = frame.lineTop(deco.first);
        const F32 bottom = frame.lineBottom(deco.last);
        pools.boxes.push_back({
            .rect = {frame.rect.x + frame.bodyPad - frame.scrollX, top, frame.body * kQuoteBarRatio, bottom - top},
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
            .fontSize = frame.body * document.scaleAt(deco.first),
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

    void addStatTiles(const DecorationFrame& frame, const Deco& deco, DecorationPools& pools) const {
        const StatGrid& grid = document.stats[deco.table];
        const F32 pad = frame.body * kStatPadEmRatio;
        const F32 left = frame.rect.x + frame.bodyPad - frame.scrollX;
        for (U64 start = 0; start < grid.tiles.size(); start += grid.columns) {
            const U64 end = std::min<U64>(grid.tiles.size(), start + grid.columns);
            F32 top = std::numeric_limits<F32>::max();
            F32 bottom = std::numeric_limits<F32>::lowest();
            for (U64 i = start; i < end; ++i) {
                top = std::min(top, frame.lineTop(grid.tiles[i].first));
                bottom = std::max(bottom, frame.lineBottom(grid.tiles[i].last));
            }
            top -= pad;
            bottom += pad;
            if (frame.clip.has_value() && bottom <= frame.clip->y) {
                continue;
            }
            if (frame.clip.has_value() && top >= frame.clip->bottom()) {
                break;
            }
            for (U64 i = start; i < end; ++i) {
                const auto& tile = grid.tiles[i];
                const Rect rect = {left + tile.x, top, grid.tileWidth, bottom - top};
                if (tile.tone.has_value()) {
                    auto& instances = pools.callouts[tile.tone.value()];
                    if (instances.size() >= kMaxCallouts) {
                        continue;
                    }
                    auto fill = frame.callout[tile.tone.value()];
                    fill = WithAlpha(fill, kCalloutFillAlpha);
                    instances.push_back({.rect = rect, .visible = frame.on, .backgroundColor = fill});
                } else if (pools.codeBoxes.size() < kMaxDecorations) {
                    pools.codeBoxes.push_back({.rect = rect, .visible = frame.on, .backgroundColor = frame.codeBlock});
                }
            }
        }
    }

    static bool offscreen(const DecorationFrame& frame, const Deco& deco) {
        if (!frame.clip.has_value() || deco.kind == Deco::Kind::Table || deco.kind == Deco::Kind::Stats) {
            return false;
        }
        const F32 pad = frame.body;
        return frame.lineBottom(deco.last) + pad + deco.tail <= frame.clip->y ||
               frame.lineTop(deco.first) - pad >= frame.clip->bottom();
    }

    DecorationPools buildDecorations(const DecorationFrame& frame) const {
        DecorationPools pools;
        for (const auto& deco : document.decos) {
            if (pools.full()) {
                break;
            }
            if (offscreen(frame, deco)) {
                continue;
            }
            DecorationFrame local = frame;
            local.bodyPad += deco.inset;
            local.bodyPadRight += deco.inset;
            switch (deco.kind) {
                case Deco::Kind::Rule: addRule(local, deco, pools); break;
                case Deco::Kind::Underline: addUnderline(local, deco, pools); break;
                case Deco::Kind::Code: addCodeBlock(local, deco, pools); break;
                case Deco::Kind::Quote: addQuoteBar(local, deco, pools); break;
                case Deco::Kind::ListItem: addListMarker(local, deco, pools); break;
                case Deco::Kind::Table: addTableChrome(local, deco, pools); break;
                case Deco::Kind::Callout: addCallout(local, deco, pools); break;
                case Deco::Kind::Stats: addStatTiles(local, deco, pools); break;
                case Deco::Kind::Details: addDetails(local, deco, pools); break;
            }
        }
        return pools;
    }

    void uploadDecorations(const DecorationFrame& frame, DecorationPools&& pools) {
        for (U64 tone = 0; tone < kCalloutTones; ++tone) {
            auto border = frame.callout[tone];
            border = WithAlpha(border, kCalloutBorderAlpha);
            calloutChrome[tone].update({
                .instances = std::move(pools.callouts[tone]),
                .clip = frame.clip,
                .cornerRadius = std::max(1.0f, frame.body * kCalloutRadiusRatio),
                .borderWidth = std::max(1.0f, std::round(frame.body * kCalloutBorderRatio * 2.0f) * 0.5f),
                .borderColor = border,
                .strokeFringe = true,
                .capacity = kMaxCallouts,
            });
        }
        codeChrome.update({
            .instances = std::move(pools.codeBoxes),
            .clip = frame.clip,
            .cornerRadius = std::max(1.0f, frame.body * kCalloutRadiusRatio),
            .borderWidth = frame.codeBorder,
            .borderColor = frame.codeBlockBorder,
            .strokeFringe = true,
            .capacity = kMaxDecorations,
        });
        tableChrome.update({
            .instances = std::move(pools.tableBoxes),
            .clip = frame.clip,
            .capacity = kMaxDecorations,
        });
        decorations.update({
            .instances = std::move(pools.boxes),
            .clip = frame.clip,
            .cornerRadius = std::max(1.0f, frame.body * kDecorationCornerRatio),
            .capacity = kMaxDecorations,
        });
        markers.update({
            .instances = std::move(pools.markers),
            .clip = frame.clip,
            .fontName = kCodeFont,
            .maxCharacters = kMaxMarkerCharacters,
            .capacity = kMaxDecorations,
        });
        headers.update({
            .instances = std::move(pools.headers),
            .clip = frame.clip,
            .fontName = kCodeFont,
            .maxCharacters = kMaxHeaderCharacters,
            .capacity = kMaxDecorations,
        });
        targets = std::move(pools.targets);
    }

    std::optional<U64> copyAt(const std::optional<Extent2D<F32>>& point, const Rect& visible) const {
        if (!point.has_value()) {
            return std::nullopt;
        }
        const auto target = targetAt(*point, visible);
        return target && target->copy ? std::optional<U64>(target->line) : std::nullopt;
    }

    std::optional<HitTarget> targetAt(const Extent2D<F32>& point, const Rect& visible) const {
        if (!visible.contains(point.x, point.y)) {
            return std::nullopt;
        }
        for (const auto& target : targets) {
            if (target.rect.contains(point.x, point.y)) {
                return target;
            }
        }
        return std::nullopt;
    }

    void activate(const HitTarget& target) {
        if (target.copy) {
            if (const Deco* code = codeHeaderAt(target.line)) {
                SetClipboardText(textMetrics.window, code->text);
            }
            return;
        }
        if (const auto detail = summaryAt(target.line)) {
            details[*detail].open = !details[*detail].open;
            if (config.onDetailsToggle) {
                config.detailsOpen.clear();
                for (const auto& state : details) {
                    config.detailsOpen.push_back(state.open);
                }
                config.onDetailsToggle(config.detailsOpen);
            }
            rebuild();
            syncGrid();
            owner->invalidate(Dirty::Layout);
        }
    }
};

TextMarkdown::TextMarkdown() {
    this->impl = std::make_unique<Impl>();
    this->impl->owner = this;
    setClipsChildren(true);
    for (auto& chrome : this->impl->calloutChrome) {
        addChild(chrome);
    }
    addChild(this->impl->codeChrome);
    addChild(this->impl->tableChrome);
    addChild(this->impl->decorations);
    addChild(this->impl->grid);
    addChild(this->impl->markers);
    addChild(this->impl->headers);
    this->impl->syncGrid();
}

TextMarkdown::~TextMarkdown() = default;

void TextMarkdown::update(Config config) {
    if (config != impl->config) {
        invalidate(Dirty::Layout);
        impl->gridConfigDirty = true;
    }
    const bool changed = !impl->parsed ||
                         impl->config.id != config.id ||
                         impl->config.value != config.value ||
                         impl->config.hero != config.hero ||
                         impl->config.softBreaks != config.softBreaks ||
                         impl->config.detailsOpen != config.detailsOpen ||
                         impl->config.fontSize != config.fontSize ||
                         impl->config.styleScales != config.styleScales;
    const bool widthInputsChanged = impl->config.padding != config.padding ||
                                    impl->config.styleFonts != config.styleFonts ||
                                    impl->config.styleScales != config.styleScales ||
                                    impl->config.styleBackgroundColorKeys != config.styleBackgroundColorKeys;
    if (impl->config.id != config.id) {
        impl->details.clear();
    }
    impl->config = std::move(config);
    if (widthInputsChanged) {
        impl->tablesLayoutValid = false;
    }
    if (changed) {
        impl->rebuild();
        impl->parsed = true;
    }
    if (changed || widthInputsChanged) {
        impl->syncGrid();
    }
}

F32 TextMarkdown::naturalWidth() const {
    return impl->grid.naturalWidth();
}

const TextGrid::Metrics& TextMarkdown::metrics() const {
    return impl->grid.metrics();
}

const std::vector<TextMarkdown::Heading>& TextMarkdown::headings() const {
    return impl->document.headings;
}

void TextMarkdown::scrollToLine(U64 line) {
    impl->grid.scrollToLine(line);
}

Extent2D<F32> TextMarkdown::measure(const Context& ctx, Extent2D<F32> available) {
    impl->textMetrics.setWindow(ctx.render);
    impl->syncGrid();
    return FitExtent(measureChild(impl->grid, ctx, available), available);
}

void TextMarkdown::layout(const Context& ctx) {
    const Rect bounds = frame();
    const Rect clipPixel = Intersect(bounds, clip());

    impl->textMetrics.setWindow(ctx.render);
    impl->syncGrid();
    layoutChild(ctx, impl->grid, bounds);

    const auto decoFrame = impl->decorationFrame(ctx, bounds, clipPixel);
    impl->uploadDecorations(decoFrame, impl->buildDecorations(decoFrame));
    std::optional<Extent2D<F32>> pointer;
    if (impl->pointer.has_value()) {
        const auto offset = translation();
        pointer = Extent2D<F32>{impl->pointer->x - offset.x, impl->pointer->y - offset.y};
    }
    if (const auto hovered = impl->copyAt(pointer, clipPixel); hovered != impl->hoveredCopy) {
        impl->hoveredCopy = hovered;
        impl->uploadDecorations(decoFrame, impl->buildDecorations(decoFrame));
    }

    for (auto& chrome : impl->calloutChrome) {
        layoutChild(ctx, chrome, bounds);
    }
    layoutChild(ctx, impl->codeChrome, bounds);
    layoutChild(ctx, impl->tableChrome, bounds);
    layoutChild(ctx, impl->decorations, bounds);
    layoutChild(ctx, impl->markers, bounds);
    layoutChild(ctx, impl->headers, bounds);
}

bool TextMarkdown::mouseEvent(const MouseEvent& event) {
    const Rect visible = Intersect(frame(), clip());
    switch (event.type) {
        case MouseEventType::Click:
            if (event.button == MouseButton::Left) {
                if (const auto target = impl->targetAt(event.position, visible)) {
                    impl->pressed = target;
                    return true;
                }
            }
            break;
        case MouseEventType::Release:
            if (const auto target = std::exchange(impl->pressed, std::nullopt)) {
                if (impl->targetAt(event.position, visible) == target) {
                    impl->activate(*target);
                }
                return true;
            }
            break;
        case MouseEventType::Move: {
            const auto offset = translation();
            impl->pointer = Extent2D<F32>{event.position.x + offset.x, event.position.y + offset.y};
            const auto hovered = impl->copyAt(event.position, visible);
            if (hovered != impl->hoveredCopy) {
                impl->hoveredCopy = hovered;
                invalidate(Dirty::Layout);
            }
            break;
        }
        case MouseEventType::Cancel:
        case MouseEventType::Leave:
            impl->pressed.reset();
            impl->pointer.reset();
            if (std::exchange(impl->hoveredCopy, std::nullopt).has_value()) {
                invalidate(Dirty::Layout);
            }
            break;
        default:
            break;
    }
    return mouseEventChildren(event);
}

std::optional<SurfaceCursor> TextMarkdown::cursorOverChildren(const Extent2D<F32>& point) const {
    if (impl->targetAt(point, Intersect(frame(), clip())).has_value()) {
        return SurfaceCursor::Hand;
    }
    return std::nullopt;
}

}  // namespace Jetstream::Sakura
