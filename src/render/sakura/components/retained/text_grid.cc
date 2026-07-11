#include <jetstream/render/sakura/components/retained/text_grid.hh>

#include <jetstream/render/sakura/clipboard.hh>
#include <jetstream/render/sakura/components/retained/box.hh>
#include <jetstream/render/sakura/components/retained/label.hh>
#include <jetstream/render/sakura/components/retained/scroll_view.hh>

#include "../../helpers.hh"
#include "../../state.hh"
#include "../../retained/helpers.hh"
#include "../../retained/text_metrics.hh"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace Jetstream::Sakura::Retained {

namespace {

constexpr F32 kReferenceFontSize = 15.0f;
constexpr F32 kLineHeightFontRatio = 1.15f;
constexpr F32 kPaddingFontRatio = 6.0f / kReferenceFontSize;
constexpr F32 kScrollbarThicknessFontRatio = 6.0f / kReferenceFontSize;
constexpr F32 kScrollbarMarginFontRatio = 4.0f / kReferenceFontSize;
constexpr F32 kSeparatorWidthFontRatio = 1.0f / kReferenceFontSize;
constexpr F32 kCursorWidthFontRatio = 2.0f / kReferenceFontSize;
constexpr F32 kCursorScrollMarginFontRatio = 24.0f / kReferenceFontSize;
constexpr F32 kLineNumberLeftPaddingCharacters = 1.0f;
constexpr F32 kLineNumberRightPaddingCharacters = 1.5f;
constexpr U64 kLineNumberDigits = 4;
constexpr U64 kMaxLineNumberCharacters = 6;
constexpr F32 kWheelScrollLines = 3.0f;
constexpr F32 kDragScrollMarginFontRatio = 36.0f / kReferenceFontSize;
constexpr F32 kDragScrollMaxLines = 0.85f;
constexpr F32 kFallbackAdvanceFontRatio = 0.5f;
constexpr U64 kVisibleLineCapacity = 64;
constexpr U64 kTextSegmentCharacterCapacity = 128;
constexpr U64 kSelectionMatchCapacity = 128;
constexpr U64 kMaxUndoHistory = 128;
constexpr U64 kTabSize = 4;
constexpr F64 kCursorBlinkPeriod = 1.0;
constexpr F64 kCursorBlinkOnDuration = 0.55;
constexpr std::string_view kAutoPairOpenings = "([{\"'";
constexpr std::string_view kAutoPairClosings = ")]}\"'";

std::vector<std::string> SplitLines(const std::string& value) {
    std::vector<std::string> lines;
    std::string::size_type start = 0;
    while (true) {
        const auto end = value.find('\n', start);
        if (end == std::string::npos) {
            lines.push_back(value.substr(start));
            break;
        }
        lines.push_back(value.substr(start, end - start));
        start = end + 1;
    }
    return lines;
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

bool IsWordChar(char character) {
    return std::isalnum(static_cast<unsigned char>(character)) || character == '_';
}

bool IsBracketPair(char opening, char closing) {
    return (opening == '(' && closing == ')') ||
           (opening == '[' && closing == ']') ||
           (opening == '{' && closing == '}');
}

I32 CharacterClass(char character) {
    if (std::isspace(static_cast<unsigned char>(character))) {
        return 0;
    }
    if (IsWordChar(character)) {
        return 1;
    }
    return 2;
}

U64 LeadingWhitespaceColumn(const std::string& line) {
    U64 column = 0;
    while (column < line.size() && (line[column] == ' ' || line[column] == '\t')) {
        ++column;
    }
    return column;
}

std::vector<std::pair<U64, U64>> WrapSegments(std::string_view line, U64 cols, bool word) {
    std::vector<std::pair<U64, U64>> segments;
    const U64 len = line.size();
    if (cols == 0 || len <= cols) {
        segments.emplace_back(0, len);
        return segments;
    }
    if (!word) {
        for (U64 start = 0; start < len; start += cols) {
            segments.emplace_back(start, std::min(len, start + cols));
        }
        return segments;
    }
    U64 pos = 0;
    while (pos < len) {
        const U64 limit = std::min(len, pos + cols);
        U64 end = limit;
        if (limit < len) {
            U64 brk = pos;
            for (U64 i = limit; i > pos; --i) {
                if (line[i - 1] == ' ' || line[i - 1] == '\t') {
                    brk = i;
                    break;
                }
            }
            end = brk > pos ? brk : limit;
        }
        segments.emplace_back(pos, end);
        pos = end;
    }
    return segments;
}

std::string& ActiveTextGridId() {
    static std::string id;
    return id;
}

}  // namespace

struct TextGrid::Impl {
    using Position = TextGrid::Position;

    struct Snapshot {
        std::vector<std::string> lines;
        Position cursor;
        Position anchor;
        bool selectionActive = false;
    };

    struct ResolvedTheme {
        ColorRGBA<F32> background = {0.0f, 0.0f, 0.0f, 0.0f};
        ColorRGBA<F32> text = {1.0f, 1.0f, 1.0f, 1.0f};
        ColorRGBA<F32> lineNumber = {0.5f, 0.5f, 0.5f, 1.0f};
        ColorRGBA<F32> gutterSeparator = {0.3f, 0.3f, 0.3f, 1.0f};
        ColorRGBA<F32> selection = {0.3f, 0.4f, 0.6f, 0.5f};
        ColorRGBA<F32> selectionMatch = {0.3f, 0.4f, 0.6f, 0.3f};
        ColorRGBA<F32> activeLine = {1.0f, 1.0f, 1.0f, 0.04f};
        ColorRGBA<F32> cursor = {1.0f, 1.0f, 1.0f, 1.0f};
        ColorRGBA<F32> scrollbarTrack = {1.0f, 1.0f, 1.0f, 0.08f};
        ColorRGBA<F32> scrollbarThumb = {1.0f, 1.0f, 1.0f, 0.25f};
        std::vector<ColorRGBA<F32>> styleColors;
        std::vector<ColorRGBA<F32>> styleBackgrounds;
    };

    Config config;
    ResolvedTheme theme;

    ScrollView scroll;
    Box backgroundBox;
    Box activeLineBox;
    Box selectionMatchBox;
    Box selectionBox;
    Box styleBackgroundBox;
    Box gutterBox;
    Box cursorBox;
    Label codeLabels;
    Label numberLabels;
    std::vector<std::unique_ptr<Label>> extraFontLabels;
    std::vector<std::string> extraFontNames;
    std::function<void(Component&)> addChild;
    mutable TextMetrics textMetrics;

    Rect rect;
    std::optional<Rect> clip;
    Metrics storedMetrics;
    bool hovered = false;
    bool active = false;
    bool windowFocused = false;

    std::vector<std::string> lines = {""};
    F32 currentScrollX = 0.0f;
    F32 currentScrollY = 0.0f;
    bool stickToBottomPending = false;
    F32 fontSizePixels = kReferenceFontSize;

    bool focused = false;
    bool mouseSelecting = false;
    Position cursor;
    Position selectionAnchor;
    bool selectionActive = false;
    std::optional<U64> preferredColumn;
    std::vector<Snapshot> undoStack;
    std::vector<Snapshot> redoStack;
    U64 contentRevision = 0;
    F64 blinkBase = 0.0;
    bool lastBlinkOn = false;

    F32 lineHeightPixels() const { return std::max(1.0f, fontSizePixels * kLineHeightFontRatio); }
    F32 paddingPixels() const { return fontSizePixels * kPaddingFontRatio; }
    F32 contentFontSize() const { return fontSizePixels * config.fontScale; }

    bool variableMetrics() const { return !config.lineScale.empty(); }
    F32 lineScaleAt(U64 line) const {
        return line < config.lineScale.size() ? config.lineScale[line] : 1.0f;
    }
    F32 lineTopGapAt(U64 line) const {
        return line < config.lineTopGap.size() ? config.lineTopGap[line] : 0.0f;
    }
    F32 lineIndentAt(U64 line) const {
        return line < config.lineIndent.size() ? config.lineIndent[line] : 0.0f;
    }
    F32 lineGlyphSize(U64 line) const { return contentFontSize() * lineScaleAt(line); }
    F32 lineHeightAt(U64 line) const {
        return std::max(1.0f, fontSizePixels * lineScaleAt(line) * kLineHeightFontRatio);
    }
    F32 lineTextLeftPixels(U64 line) const { return textLeftPixels() + lineIndentAt(line); }

    F32 rowTopContent(U64 visualRow) const {
        if (variableMetrics()) {
            return visualRow < visualRows.size() ? visualRows[visualRow].top : visualContentHeight;
        }
        return static_cast<F32>(visualRow) * lineHeightPixels();
    }
    F32 rowHeightPixels(U64 visualRow) const {
        if (variableMetrics()) {
            return visualRow < visualRows.size() ? visualRows[visualRow].height : lineHeightPixels();
        }
        return lineHeightPixels();
    }
    F32 visibleTopContentY() const {
        const F32 clipTop = clip.has_value()
            ? std::max(rect.y, clip->y)
            : rect.y;
        return std::max(0.0f, (clipTop - viewportTopPixels()) + currentScrollY);
    }
    U64 firstVisibleVisualRow() const {
        if (variableMetrics()) {
            return visualRowAtContentY(visibleTopContentY());
        }
        return static_cast<U64>(std::max(0.0f, std::floor(visibleTopContentY() / lineHeightPixels())));
    }

    F32 sourceLineTop(U64 line) const {
        ensureVisualRows();
        if (line < visualRowStartIndex.size()) {
            return rowTopPixels(visualRowStartIndex[line]);
        }
        return viewportTopPixels() + rowTopContent(visualRows.size()) - currentScrollY;
    }
    F32 sourceLineHeight(U64 line) const {
        ensureVisualRows();
        if (line >= visualRowStartIndex.size()) {
            return 0.0f;
        }
        const U64 first = visualRowStartIndex[line];
        const U64 last = line + 1 < visualRowStartIndex.size()
            ? visualRowStartIndex[line + 1]
            : visualRows.size();
        F32 h = 0.0f;
        for (U64 r = first; r < last; ++r) {
            h += rowHeightPixels(r);
        }
        return h;
    }
    U64 visualRowAtContentY(F32 y) const {
        if (visualRows.empty()) {
            return 0;
        }
        U64 lo = 0;
        U64 hi = visualRows.size();
        while (lo + 1 < hi) {
            const U64 mid = (lo + hi) / 2;
            if (visualRows[mid].top <= y) {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        return lo;
    }

    F32 characterAdvancePixels() const {
        const F32 advance = textMetrics.measure(config.fontName, "0", contentFontSize());
        return advance > 0.0f ? advance : contentFontSize() * kFallbackAdvanceFontRatio;
    }

    bool fontsReady() const {
        const F32 fontSize = contentFontSize();
        if (textMetrics.measure(config.fontName, "0", fontSize) <= 0.0f) {
            return false;
        }
        for (const auto& fontName : extraFontNames) {
            if (textMetrics.measure(fontName, "0", fontSize) <= 0.0f) {
                return false;
            }
        }
        return true;
    }

    mutable std::vector<std::vector<F32>> linePrefixes;
    mutable bool linePrefixesValid = false;
    mutable U64 linePrefixesRevision = static_cast<U64>(-1);
    mutable F32 linePrefixesFontSize = -1.0f;

    void ensureLinePrefixes() const {
        const F32 fontSize = contentFontSize();
        if (linePrefixesValid && linePrefixesRevision == contentRevision &&
            linePrefixesFontSize == fontSize) {
            return;
        }
        const bool ready = fontsReady();

        static const std::vector<std::vector<StyleId>> kNoStyles;
        static const std::vector<StyleId> kNoLineStyles;
        const bool multiFont = !extraFontNames.empty() && config.styler;
        const auto& styles = multiFont ? config.styler(lines, contentRevision) : kNoStyles;

        linePrefixes.assign(lines.size(), {});
        for (U64 i = 0; i < lines.size(); ++i) {
            const auto& line = lines[i];
            auto& prefix = linePrefixes[i];
            prefix.resize(line.size() + 1, 0.0f);
            const F32 lineSize = lineGlyphSize(i);
            if (!multiFont) {
                const auto adv = textMetrics.advances(config.fontName, line, lineSize);
                for (U64 c = 0; c < line.size(); ++c) {
                    prefix[c + 1] = prefix[c] + (c < adv.size() ? adv[c] : 0.0f);
                }
                continue;
            }
            const auto& lineStyles = i < styles.size() ? styles[i] : kNoLineStyles;
            U64 c = 0;
            while (c < line.size()) {
                const std::string font = fontForStyle(c < lineStyles.size() ? lineStyles[c] : 0);
                U64 e = c + 1;
                while (e < line.size() && fontForStyle(e < lineStyles.size() ? lineStyles[e] : 0) == font) {
                    ++e;
                }
                const auto adv = textMetrics.advances(font.empty() ? config.fontName : font,
                                                      line.substr(c, e - c), lineSize);
                for (U64 k = c; k < e; ++k) {
                    prefix[k + 1] = prefix[k] + (k - c < adv.size() ? adv[k - c] : 0.0f);
                }
                c = e;
            }
        }
        if (ready) {
            linePrefixesValid = true;
            linePrefixesRevision = contentRevision;
            linePrefixesFontSize = fontSize;
        }
    }

    F32 columnOffsetPixels(U64 lineIndex, U64 column) const {
        if (lineIndex >= lines.size()) {
            return 0.0f;
        }
        const U64 col = std::min<U64>(column, lines[lineIndex].size());
        if (config.monospace) {
            return static_cast<F32>(col) * characterAdvancePixels();
        }
        ensureLinePrefixes();
        const auto& prefix = linePrefixes[lineIndex];
        return col < prefix.size() ? prefix[col] : (prefix.empty() ? 0.0f : prefix.back());
    }

    F32 columnXInRow(U64 lineIndex, U64 rowStart, U64 column) const {
        return columnOffsetPixels(lineIndex, column) - columnOffsetPixels(lineIndex, rowStart);
    }

    F32 lineWidthPixels(U64 lineIndex) const {
        return lineIndex < lines.size() ? columnOffsetPixels(lineIndex, lines[lineIndex].size()) : 0.0f;
    }

    U64 columnAtOffsetPixels(U64 lineIndex, F32 x) const {
        if (lineIndex >= lines.size()) {
            return 0;
        }
        const U64 len = lines[lineIndex].size();
        if (config.monospace) {
            const F32 advance = std::max(1.0f, characterAdvancePixels());
            const I64 col = static_cast<I64>(x / advance + 0.5f);
            return static_cast<U64>(std::clamp<I64>(col, 0, static_cast<I64>(len)));
        }
        ensureLinePrefixes();
        const auto& prefix = linePrefixes[lineIndex];
        if (prefix.size() <= 1 || x <= 0.0f) {
            return 0;
        }
        if (x >= prefix.back()) {
            return len;
        }
        const U64 hi = static_cast<U64>(std::upper_bound(prefix.begin(), prefix.end(), x) - prefix.begin());
        const U64 lo = hi - 1;
        return (x - prefix[lo] <= prefix[hi] - x) ? lo : hi;
    }

    F32 gutterWidthPixels() const {
        if (!config.lineNumbers) {
            return 0.0f;
        }
        return (kLineNumberLeftPaddingCharacters + static_cast<F32>(kLineNumberDigits) +
                kLineNumberRightPaddingCharacters) * characterAdvancePixels();
    }
    F32 lineNumberRightPixels() const {
        return (kLineNumberLeftPaddingCharacters + static_cast<F32>(kLineNumberDigits)) *
               characterAdvancePixels();
    }
    F32 textLeftPixels() const {
        return config.lineNumbers ? rect.x + gutterWidthPixels() : rect.x + paddingPixels();
    }
    F32 textClipRightPixels() const { return rect.right() - paddingPixels(); }
    F32 textViewportWidthPixels() const {
        return std::max(1.0f, textClipRightPixels() - textLeftPixels());
    }
    F32 viewportTopPixels() const { return rect.y; }
    struct VisualRow {
        U64 line = 0;
        U64 start = 0;
        U64 end = 0;
        F32 top = 0.0f;
        F32 height = 0.0f;
    };
    mutable std::vector<VisualRow> visualRows;
    mutable std::vector<U64> visualRowStartIndex;
    mutable F32 visualContentHeight = 0.0f;
    mutable bool visualRowsValid = false;
    mutable U64 visualRowsRevision = 0;
    mutable Wrap visualRowsWrap = Wrap::None;
    mutable U64 visualRowsWrapColumns = 0;

    U64 wrapColumns() const {
        if (config.wrap == Wrap::None) {
            return 0;
        }
        const F32 advance = std::max(1.0f, characterAdvancePixels());
        return std::max<U64>(1, static_cast<U64>(std::floor(textViewportWidthPixels() / advance)));
    }

    std::vector<std::pair<U64, U64>> wrapSegmentsProportional(U64 lineIndex, F32 maxWidth, bool word) const {
        std::vector<std::pair<U64, U64>> segments;
        const U64 len = lines[lineIndex].size();
        if (len == 0 || maxWidth <= 0.0f) {
            segments.emplace_back(0, len);
            return segments;
        }
        ensureLinePrefixes();
        const auto& prefix = linePrefixes[lineIndex];
        const auto& text = lines[lineIndex];
        U64 pos = 0;
        while (pos < len) {
            const F32 budget = prefix[pos] + maxWidth;
            U64 fitEnd = static_cast<U64>(std::upper_bound(prefix.begin(), prefix.end(), budget) - prefix.begin());
            if (fitEnd > 0) {
                fitEnd -= 1;
            }
            fitEnd = std::clamp<U64>(fitEnd, pos + 1, len);
            U64 end = fitEnd;
            if (word && fitEnd < len) {
                U64 brk = pos;
                for (U64 i = fitEnd; i > pos; --i) {
                    if (text[i - 1] == ' ' || text[i - 1] == '\t') {
                        brk = i;
                        break;
                    }
                }
                end = brk > pos ? brk : fitEnd;
            }
            segments.emplace_back(pos, end);
            pos = end;
        }
        if (segments.empty()) {
            segments.emplace_back(0, len);
        }
        return segments;
    }

    void ensureVisualRows() const {
        const U64 wrapCols = wrapColumns();
        if (visualRowsValid && visualRowsRevision == contentRevision &&
            visualRowsWrap == config.wrap && visualRowsWrapColumns == wrapCols) {
            return;
        }
        visualRows.clear();
        visualRowStartIndex.assign(lines.size(), 0);
        const F32 trailingMargin = 0.5f * characterAdvancePixels();
        F32 y = 0.0f;
        for (U64 line = 0; line < lines.size(); ++line) {
            y += lineTopGapAt(line);
            visualRowStartIndex[line] = visualRows.size();
            const auto& text = lines[line];
            const U64 len = text.size();
            const F32 h = lineHeightAt(line);
            const auto push = [&](U64 start, U64 end) {
                visualRows.push_back({line, start, end, y, h});
                y += h;
            };
            if (config.wrap == Wrap::None) {
                push(0, len);
            } else if (config.monospace) {
                for (const auto& [start, end] : WrapSegments(text, wrapCols, config.wrap == Wrap::Word)) {
                    push(start, end);
                }
            } else {
                const F32 wrapWidth = std::max(
                    1.0f, textViewportWidthPixels() - lineIndentAt(line) - trailingMargin);
                for (const auto& [start, end] : wrapSegmentsProportional(line, wrapWidth, config.wrap == Wrap::Word)) {
                    push(start, end);
                }
            }
        }
        if (visualRows.empty()) {
            visualRows.push_back({0, 0, 0, 0.0f, lineHeightAt(0)});
            y = lineHeightAt(0);
        }
        visualContentHeight = y;
        if (config.monospace || config.wrap == Wrap::None || fontsReady()) {
            visualRowsValid = true;
            visualRowsRevision = contentRevision;
            visualRowsWrap = config.wrap;
            visualRowsWrapColumns = wrapCols;
        }
    }

    U64 visualRowCount() const {
        ensureVisualRows();
        return visualRows.size();
    }

    U64 visualRowForPosition(Position position) const {
        ensureVisualRows();
        if (position.line >= visualRowStartIndex.size()) {
            return visualRows.empty() ? 0 : visualRows.size() - 1;
        }
        const U64 base = visualRowStartIndex[position.line];
        const U64 next = position.line + 1 < visualRowStartIndex.size()
                             ? visualRowStartIndex[position.line + 1]
                             : visualRows.size();
        for (U64 r = base; r < next; ++r) {
            if (position.column >= visualRows[r].start && position.column < visualRows[r].end) {
                return r;
            }
        }
        return next > base ? next - 1 : base;
    }

    F32 rowTopPixels(U64 visualRow) const {
        return viewportTopPixels() + rowTopContent(visualRow) - currentScrollY;
    }

    F32 editorBottomPaddingPixels() const {
        return config.editable ? std::max(0.0f, rect.height - lineHeightPixels()) : 0.0f;
    }
    F32 textContentHeightPixels() const {
        ensureVisualRows();
        return variableMetrics()
            ? visualContentHeight
            : static_cast<F32>(visualRows.size()) * lineHeightPixels();
    }
    F32 contentHeightPixels() const {
        return textContentHeightPixels() + editorBottomPaddingPixels();
    }
    F32 maxLineAdvancePixels() const {
        if (config.wrap != Wrap::None) {
            return 0.0f;
        }
        F32 widest = 0.0f;
        for (U64 i = 0; i < lines.size(); ++i) {
            widest = std::max(widest, lineWidthPixels(i));
        }
        return widest;
    }
    F32 contentWidthPixels() const {
        return (textLeftPixels() - rect.x) + maxLineAdvancePixels() + paddingPixels();
    }
    F32 measuredContentWidthPixels() const {
        ensureVisualRows();
        F32 widest = 0.0f;
        for (const auto& row : visualRows) {
            widest = std::max(widest, columnXInRow(row.line, row.start, row.end) + lineIndentAt(row.line));
        }
        return (textLeftPixels() - rect.x) + widest + paddingPixels();
    }
    Metrics computeMetrics() const {
        Metrics out;
        out.contentHeight = textContentHeightPixels();
        out.sourceLines.resize(lines.size());
        for (U64 line = 0; line < lines.size(); ++line) {
            out.sourceLines[line] = {
                .top = sourceLineTop(line),
                .height = sourceLineHeight(line),
            };
        }
        return out;
    }
    void notifyLayout() {
        storedMetrics = computeMetrics();
        if (config.onLayout) {
            config.onLayout(storedMetrics);
        }
    }
    U64 pageStepLines() const {
        const F32 rows = rect.height / lineHeightPixels();
        return static_cast<U64>(std::max(1.0f, rows - 1.0f));
    }

    void resolveTheme(const Context& ctx) {
        theme.background = ctx.color(config.backgroundColorKey);
        theme.text = ctx.color(config.textColorKey);
        theme.lineNumber = ctx.color(config.lineNumberColorKey);
        theme.gutterSeparator = ctx.color(config.gutterSeparatorColorKey);
        theme.selection = ctx.color(config.selectionColorKey);
        theme.selectionMatch = ctx.color(config.selectionMatchColorKey);
        theme.activeLine = ctx.color(config.activeLineColorKey);
        theme.cursor = ctx.color(config.cursorColorKey);
        theme.scrollbarTrack = ctx.color(config.scrollbarTrackColorKey);
        theme.scrollbarThumb = ctx.color(config.scrollbarThumbColorKey);

        theme.styleColors.clear();
        theme.styleColors.reserve(config.styleColorKeys.size());
        for (const auto& key : config.styleColorKeys) {
            theme.styleColors.push_back(key.empty() ? theme.text : ctx.color(key));
        }

        theme.styleBackgrounds.clear();
        theme.styleBackgrounds.reserve(config.styleBackgroundColorKeys.size());
        for (const auto& key : config.styleBackgroundColorKeys) {
            theme.styleBackgrounds.push_back(ctx.color(key.empty() ? "transparent" : key));
        }
    }

    ColorRGBA<F32> colorForStyle(StyleId id) const {
        if (id == 0 || id > theme.styleColors.size()) {
            return theme.text;
        }
        return theme.styleColors[id - 1];
    }

    std::string fontForStyle(StyleId id) const {
        if (id == 0 || id > config.styleFonts.size()) {
            return config.fontName;
        }
        const auto& font = config.styleFonts[id - 1];
        return font.empty() ? config.fontName : font;
    }

    ColorRGBA<F32> backgroundForStyle(StyleId id) const {
        if (id == 0 || id > theme.styleBackgrounds.size()) {
            return {0.0f, 0.0f, 0.0f, 0.0f};
        }
        return theme.styleBackgrounds[id - 1];
    }

    void ensureFontPools() {
        std::vector<std::string> wanted;
        for (const auto& font : config.styleFonts) {
            if (font.empty() || font == config.fontName) {
                continue;
            }
            if (std::find(wanted.begin(), wanted.end(), font) == wanted.end()) {
                wanted.push_back(font);
            }
        }
        if (wanted == extraFontNames) {
            return;
        }
        extraFontNames = std::move(wanted);
        while (extraFontLabels.size() < extraFontNames.size()) {
            extraFontLabels.push_back(std::make_unique<Label>());
            if (addChild) {
                addChild(*extraFontLabels.back());
            }
        }
        linePrefixesValid = false;
    }

    U64 poolIndexForFont(const std::string& font) const {
        if (!font.empty() && font != config.fontName) {
            for (U64 k = 0; k < extraFontNames.size(); ++k) {
                if (extraFontNames[k] == font) {
                    return k;
                }
            }
        }
        return static_cast<U64>(-1);
    }
    std::string textValue() const { return JoinLines(lines); }

    void clampPosition(Position& position) const {
        position.line = std::min<U64>(position.line, lines.size() - 1);
        position.column = std::min<U64>(position.column, lines[position.line].size());
    }

    bool hasSelection() const { return selectionActive && !(selectionAnchor == cursor); }

    std::pair<Position, Position> selectionRange() const {
        return positionLess(selectionAnchor, cursor) ? std::pair{selectionAnchor, cursor}
                                                     : std::pair{cursor, selectionAnchor};
    }

    static bool positionLess(const Position& a, const Position& b) {
        return a.line != b.line ? a.line < b.line : a.column < b.column;
    }

    void clearSelection() {
        selectionActive = false;
        selectionAnchor = cursor;
    }

    void notifySelect() {
        if (config.onSelect) {
            const auto [start, end] = selectionRange();
            config.onSelect(start, end);
        }
    }

    std::optional<std::string> singleLineSelectionText() const {
        if (!hasSelection()) {
            return std::nullopt;
        }
        const auto [start, end] = selectionRange();
        if (start.line != end.line || start.line >= lines.size() || end.column <= start.column) {
            return std::nullopt;
        }
        const auto text = lines[start.line].substr(start.column, end.column - start.column);
        if (text.empty() || std::all_of(text.begin(), text.end(), [](char c) {
                return std::isspace(static_cast<unsigned char>(c));
            })) {
            return std::nullopt;
        }
        return text;
    }

    std::string textInRange(const Position& start, const Position& end) const {
        if (start.line == end.line) {
            return lines[start.line].substr(start.column, end.column - start.column);
        }
        std::string value = lines[start.line].substr(start.column);
        for (U64 line = start.line + 1; line < end.line; ++line) {
            value += '\n';
            value += lines[line];
        }
        value += '\n';
        value += lines[end.line].substr(0, end.column);
        return value;
    }

    void deleteRange(const Position& start, const Position& end) {
        if (start.line == end.line) {
            lines[start.line].erase(start.column, end.column - start.column);
        } else {
            lines[start.line] = lines[start.line].substr(0, start.column) + lines[end.line].substr(end.column);
            lines.erase(lines.begin() + static_cast<I64>(start.line) + 1,
                        lines.begin() + static_cast<I64>(end.line) + 1);
        }
        cursor = start;
        clearSelection();
    }

    bool deleteSelectionIfAny() {
        if (!hasSelection()) {
            return false;
        }
        const auto [start, end] = selectionRange();
        deleteRange(start, end);
        return true;
    }

    void insertRaw(const std::string& text) {
        const auto inserted = SplitLines(text);
        auto& line = lines[cursor.line];
        const std::string tail = line.substr(cursor.column);
        line = line.substr(0, cursor.column) + inserted.front();
        if (inserted.size() == 1) {
            cursor.column += inserted.front().size();
            line += tail;
        } else {
            lines.insert(lines.begin() + static_cast<I64>(cursor.line) + 1, inserted.begin() + 1, inserted.end());
            cursor.line += inserted.size() - 1;
            cursor.column = inserted.back().size();
            lines[cursor.line] += tail;
        }
        clearSelection();
    }

    Position previousCharacterPosition() const {
        if (cursor.column > 0) {
            return {cursor.line, cursor.column - 1};
        }
        if (cursor.line > 0) {
            return {cursor.line - 1, lines[cursor.line - 1].size()};
        }
        return cursor;
    }
    Position nextCharacterPosition() const {
        if (cursor.column < lines[cursor.line].size()) {
            return {cursor.line, cursor.column + 1};
        }
        if (cursor.line + 1 < lines.size()) {
            return {cursor.line + 1, 0};
        }
        return cursor;
    }
    Position previousWordPosition() const {
        Position position = previousCharacterPosition();
        if (position == cursor) {
            return position;
        }
        const auto& line = lines[position.line];
        while (position.column > 0 && CharacterClass(line[position.column - 1]) == 0) {
            --position.column;
        }
        if (position.column > 0) {
            const I32 cls = CharacterClass(line[position.column - 1]);
            while (position.column > 0 && CharacterClass(line[position.column - 1]) == cls) {
                --position.column;
            }
        }
        return position;
    }
    Position nextWordPosition() const {
        Position position = nextCharacterPosition();
        if (position == cursor) {
            return position;
        }
        const auto& line = lines[position.line];
        U64 column = position.column;
        while (column < line.size() && CharacterClass(line[column]) == 0) {
            ++column;
        }
        if (column < line.size()) {
            const I32 cls = CharacterClass(line[column]);
            while (column < line.size() && CharacterClass(line[column]) == cls) {
                ++column;
            }
        }
        return {position.line, column};
    }
    Position lineStartPosition() const { return {cursor.line, 0}; }
    Position lineEndPosition() const { return {cursor.line, lines[cursor.line].size()}; }
    Position smartLineStartPosition() const {
        const U64 indent = LeadingWhitespaceColumn(lines[cursor.line]);
        return {cursor.line, cursor.column == indent ? 0 : indent};
    }
    Position documentStartPosition() const { return {0, 0}; }
    Position documentEndPosition() const { return {lines.size() - 1, lines.back().size()}; }

    std::optional<std::pair<Position, Position>> wordRangeAt(Position position) const {
        Position clamped = position;
        clampPosition(clamped);
        const auto& line = lines[clamped.line];
        if (line.empty()) {
            return std::nullopt;
        }
        U64 column = std::min<U64>(clamped.column, line.size() - 1);
        const I32 cls = CharacterClass(line[column]);
        U64 start = column;
        while (start > 0 && CharacterClass(line[start - 1]) == cls) {
            --start;
        }
        U64 end = column + 1;
        while (end < line.size() && CharacterClass(line[end]) == cls) {
            ++end;
        }
        return std::pair<Position, Position>{{clamped.line, start}, {clamped.line, end}};
    }

    void resetBlink() { blinkBase = ImGui::GetTime(); }

    void moveCursorTo(Position position, bool extendSelection, bool keepPreferredColumn = false) {
        clampPosition(position);
        if (extendSelection) {
            if (!selectionActive) {
                selectionAnchor = cursor;
                selectionActive = true;
            }
        } else {
            selectionActive = false;
        }
        cursor = position;
        if (!extendSelection) {
            selectionAnchor = cursor;
        }
        if (!keepPreferredColumn) {
            preferredColumn.reset();
        }
        ensureCursorVisible();
        resetBlink();
        notifySelect();
        configureScrollView();
        rebuildVisibleInstances();
    }

    void moveCursorVertically(I64 delta, bool extendSelection) {
        ensureVisualRows();
        const U64 fromRow = visualRowForPosition(cursor);
        const U64 fromStart = visualRows[fromRow].start;
        if (!preferredColumn.has_value()) {
            preferredColumn = cursor.column >= fromStart ? cursor.column - fromStart : 0;
        }
        const U64 toRow = static_cast<U64>(
            std::clamp<I64>(static_cast<I64>(fromRow) + delta, 0, static_cast<I64>(visualRows.size()) - 1));
        const auto& row = visualRows[toRow];
        const U64 column = std::min<U64>(row.start + *preferredColumn, row.end);
        moveCursorTo({row.line, column}, extendSelection, true);
    }

    void ensureCursorVisible() {
        const F32 lineHeight = lineHeightPixels();
        const F32 areaHeight = rect.height;
        const F32 cursorTop = static_cast<F32>(visualRowForPosition(cursor)) * lineHeight;
        if (cursorTop < currentScrollY) {
            currentScrollY = cursorTop;
        } else if (cursorTop + lineHeight > currentScrollY + areaHeight) {
            currentScrollY = cursorTop + lineHeight - areaHeight;
        }
        currentScrollY = std::clamp(currentScrollY, 0.0f, std::max(0.0f, contentHeightPixels() - areaHeight));

        if (config.wrap != Wrap::None) {
            currentScrollX = 0.0f;
            return;
        }
        const F32 cursorX = columnOffsetPixels(cursor.line, cursor.column);
        const F32 viewportWidth = textViewportWidthPixels();
        const F32 margin = std::min(fontSizePixels * kCursorScrollMarginFontRatio, viewportWidth * 0.5f);
        if (cursorX < currentScrollX + margin) {
            currentScrollX = std::max(0.0f, cursorX - margin);
        } else if (cursorX > currentScrollX + viewportWidth - margin) {
            currentScrollX = std::max(0.0f, cursorX - viewportWidth + margin);
        }
        currentScrollX = std::clamp(currentScrollX, 0.0f, std::max(0.0f, maxLineAdvancePixels() - viewportWidth));
    }

    Snapshot snapshot() const { return {lines, cursor, selectionAnchor, selectionActive}; }
    void restoreSnapshot(const Snapshot& state) {
        lines = state.lines;
        cursor = state.cursor;
        selectionAnchor = state.anchor;
        selectionActive = state.selectionActive;
        clampPosition(cursor);
        clampPosition(selectionAnchor);
    }
    void recordUndoState() {
        undoStack.push_back(snapshot());
        if (undoStack.size() > kMaxUndoHistory) {
            undoStack.erase(undoStack.begin());
        }
        redoStack.clear();
    }
    void commitEdit() {
        ++contentRevision;
        clampPosition(cursor);
        clampPosition(selectionAnchor);
        preferredColumn.reset();
        ensureCursorVisible();
        resetBlink();
        configureScrollView();
        rebuildVisibleInstances();
        if (config.onChange) {
            config.onChange(textValue());
        }
    }

    void insertText(const std::string& text) {
        recordUndoState();
        deleteSelectionIfAny();
        insertRaw(text);
        commitEdit();
    }

    bool suppressAutoPair(Position position) {
        if (!config.isStyleCommentOrString || !config.styler) {
            return false;
        }
        const auto& styles = config.styler(lines, contentRevision);
        if (position.line >= styles.size() || styles[position.line].empty()) {
            return false;
        }
        U64 column = position.column;
        if (column > 0) {
            --column;
        }
        column = std::min<U64>(column, styles[position.line].size() - 1);
        return config.isStyleCommentOrString(styles[position.line][column]);
    }

    bool shouldAutoPairQuote() const {
        const auto& line = lines[cursor.line];
        const bool prevWord = cursor.column > 0 && IsWordChar(line[cursor.column - 1]);
        const bool nextWord = cursor.column < line.size() && IsWordChar(line[cursor.column]);
        return !prevWord && !nextWord;
    }

    void insertAutoPair(char opening) {
        const char closing = kAutoPairClosings[kAutoPairOpenings.find(opening)];
        std::string value(1, opening);
        const bool hadSelection = hasSelection();
        if (hadSelection) {
            const auto [start, end] = selectionRange();
            value += textInRange(start, end);
        }
        value.push_back(closing);
        const Position position = hadSelection ? selectionRange().first : cursor;
        insertText(value);
        if (!hadSelection) {
            moveCursorTo({position.line, position.column + 1}, false);
        }
    }

    void insertTypedCharacter(char character) {
        const auto openingIndex = kAutoPairOpenings.find(character);
        if (hasSelection()) {
            if (openingIndex != std::string_view::npos) {
                insertAutoPair(character);
            } else {
                insertText(std::string(1, character));
            }
            return;
        }
        const auto& line = lines[cursor.line];
        if (kAutoPairClosings.find(character) != std::string_view::npos &&
            cursor.column < line.size() && line[cursor.column] == character) {
            moveCursorTo({cursor.line, cursor.column + 1}, false);
            return;
        }
        if (openingIndex != std::string_view::npos && suppressAutoPair(cursor)) {
            insertText(std::string(1, character));
            return;
        }
        const bool quote = character == '"' || character == '\'';
        if (openingIndex != std::string_view::npos && (!quote || shouldAutoPairQuote())) {
            insertAutoPair(character);
            return;
        }
        insertText(std::string(1, character));
    }

    void backspace() {
        recordUndoState();
        if (deleteSelectionIfAny()) {
            commitEdit();
            return;
        }
        auto& line = lines[cursor.line];
        if (cursor.column > 0) {
            const auto openingIndex = kAutoPairOpenings.find(line[cursor.column - 1]);
            const bool pair = openingIndex != std::string_view::npos && cursor.column < line.size() &&
                              line[cursor.column] == kAutoPairClosings[openingIndex];
            U64 leadingWhitespace = LeadingWhitespaceColumn(line);
            if (!pair && leadingWhitespace == cursor.column) {
                const U64 previousIndent = ((cursor.column - 1) / kTabSize) * kTabSize;
                line.erase(previousIndent, cursor.column - previousIndent);
                cursor.column = previousIndent;
            } else {
                line.erase(cursor.column - 1, pair ? 2 : 1);
                cursor.column -= 1;
            }
        } else if (cursor.line > 0) {
            cursor.column = lines[cursor.line - 1].size();
            lines[cursor.line - 1] += line;
            lines.erase(lines.begin() + static_cast<I64>(cursor.line));
            cursor.line -= 1;
        }
        clearSelection();
        commitEdit();
    }

    void deleteForward() {
        recordUndoState();
        if (deleteSelectionIfAny()) {
            commitEdit();
            return;
        }
        auto& line = lines[cursor.line];
        if (cursor.column < line.size()) {
            line.erase(cursor.column, 1);
        } else if (cursor.line + 1 < lines.size()) {
            line += lines[cursor.line + 1];
            lines.erase(lines.begin() + static_cast<I64>(cursor.line) + 1);
        }
        clearSelection();
        commitEdit();
    }

    void deleteToPosition(const Position& target) {
        if (target == cursor) {
            return;
        }
        recordUndoState();
        const auto [start, end] = positionLess(target, cursor) ? std::pair{target, cursor}
                                                               : std::pair{cursor, target};
        deleteRange(start, end);
        commitEdit();
    }

    void insertNewLine() {
        recordUndoState();
        deleteSelectionIfAny();
        auto& line = lines[cursor.line];
        const std::string beforeCursor = line.substr(0, cursor.column);
        const std::string tail = line.substr(cursor.column);
        std::string indent = line.substr(0, LeadingWhitespaceColumn(line));

        if (!beforeCursor.empty() && !tail.empty() &&
            IsBracketPair(beforeCursor.back(), tail.front()) && !suppressAutoPair(cursor)) {
            const std::string innerIndent = indent + std::string(kTabSize, ' ');
            line = beforeCursor;
            lines.insert(lines.begin() + static_cast<I64>(cursor.line) + 1, innerIndent);
            lines.insert(lines.begin() + static_cast<I64>(cursor.line) + 2, indent + tail);
            cursor.line += 1;
            cursor.column = innerIndent.size();
            clearSelection();
            commitEdit();
            return;
        }

        std::optional<std::string> policyIndent;
        if (config.computeNewlineIndent) {
            policyIndent = config.computeNewlineIndent(lines, cursor);
        }
        if (policyIndent.has_value()) {
            indent = *policyIndent;
        } else {
            const auto lastContent = beforeCursor.find_last_not_of(" \t");
            if (lastContent != std::string::npos && beforeCursor[lastContent] == ':') {
                indent += std::string(kTabSize, ' ');
            }
        }

        line = beforeCursor;
        lines.insert(lines.begin() + static_cast<I64>(cursor.line) + 1, indent + tail);
        cursor.line += 1;
        cursor.column = indent.size();
        clearSelection();
        commitEdit();
    }

    void indentSelection(bool unindent) {
        recordUndoState();
        const auto [start, end] = hasSelection() ? selectionRange() : std::pair{cursor, cursor};
        for (U64 lineIndex = start.line; lineIndex <= end.line; ++lineIndex) {
            auto& line = lines[lineIndex];
            I64 delta = 0;
            if (unindent) {
                U64 remove = 0;
                while (remove < kTabSize && remove < line.size() && line[remove] == ' ') {
                    ++remove;
                }
                line.erase(0, remove);
                delta = -static_cast<I64>(remove);
            } else {
                line.insert(0, std::string(kTabSize, ' '));
                delta = static_cast<I64>(kTabSize);
            }
            const auto shift = [&](Position& position) {
                if (position.line == lineIndex) {
                    position.column = static_cast<U64>(std::max<I64>(0, static_cast<I64>(position.column) + delta));
                }
            };
            shift(cursor);
            shift(selectionAnchor);
        }
        commitEdit();
    }

    void selectAll() {
        selectionAnchor = documentStartPosition();
        cursor = documentEndPosition();
        selectionActive = true;
        preferredColumn.reset();
        resetBlink();
        notifySelect();
        configureScrollView();
        rebuildVisibleInstances();
    }

    void selectWordAt(Position position) {
        const auto range = wordRangeAt(position);
        if (!range.has_value()) {
            clampPosition(position);
            moveCursorTo(position, false);
            return;
        }
        selectionAnchor = range->first;
        cursor = range->second;
        selectionActive = !(selectionAnchor == cursor);
        preferredColumn.reset();
        resetBlink();
        notifySelect();
        configureScrollView();
        rebuildVisibleInstances();
    }

    void expandSyntaxSelection() {
        if (!config.expandSelection) {
            return;
        }
        const auto range = config.expandSelection(lines, selectionAnchor, cursor);
        if (!range.has_value() || range->first == range->second) {
            return;
        }
        selectionAnchor = range->first;
        cursor = range->second;
        selectionActive = !(selectionAnchor == cursor);
        preferredColumn.reset();
        ensureCursorVisible();
        resetBlink();
        notifySelect();
        configureScrollView();
        rebuildVisibleInstances();
    }

    void copySelectionOrLine() {
        if (hasSelection()) {
            const auto [start, end] = selectionRange();
            SetClipboardText(textInRange(start, end));
        } else {
            SetClipboardText(lines[cursor.line] + "\n");
        }
    }

    void cutSelectionOrLine() {
        copySelectionOrLine();
        recordUndoState();
        if (!deleteSelectionIfAny()) {
            if (lines.size() == 1) {
                lines[0].clear();
                cursor = {0, 0};
            } else {
                lines.erase(lines.begin() + static_cast<I64>(cursor.line));
                cursor.line = std::min<U64>(cursor.line, lines.size() - 1);
                cursor.column = 0;
            }
            clearSelection();
        }
        commitEdit();
    }

    void undo() {
        if (undoStack.empty()) {
            return;
        }
        redoStack.push_back(snapshot());
        restoreSnapshot(undoStack.back());
        undoStack.pop_back();
        ++contentRevision;
        ensureCursorVisible();
        resetBlink();
        configureScrollView();
        rebuildVisibleInstances();
        if (config.onChange) {
            config.onChange(textValue());
        }
    }

    void redo() {
        if (redoStack.empty()) {
            return;
        }
        undoStack.push_back(snapshot());
        restoreSnapshot(redoStack.back());
        redoStack.pop_back();
        ++contentRevision;
        ensureCursorVisible();
        resetBlink();
        configureScrollView();
        rebuildVisibleInstances();
        if (config.onChange) {
            config.onChange(textValue());
        }
    }

    bool hasFocus() const { return focused && ActiveTextGridId() == config.id; }

    void focus() {
        const bool wasFocused = hasFocus();
        ActiveTextGridId() = config.id;
        Private::SetKeyboardInputCaptured(true);
        focused = true;
        if (!wasFocused) {
            resetBlink();
            configureScrollView();
            rebuildVisibleInstances();
        }
    }
    void clearFocus() {
        if (ActiveTextGridId() == config.id) {
            ActiveTextGridId().clear();
            Private::SetKeyboardInputCaptured(false);
        }
        if (focused) {
            focused = false;
            mouseSelecting = false;
            clearSelection();
            configureScrollView();
            rebuildVisibleInstances();
        }
    }
    void reconcileFocusOwnership() {
        if (!focused || ActiveTextGridId() == config.id) {
            return;
        }
        focused = false;
        mouseSelecting = false;
        clearSelection();
        configureScrollView();
        rebuildVisibleInstances();
    }

    void autoscrollSelection(const Extent2D<F32>& pixel) {
        const F32 margin = fontSizePixels * kDragScrollMarginFontRatio;
        const F32 maxStep = kDragScrollMaxLines * lineHeightPixels();
        F32 delta = 0.0f;
        if (pixel.y < rect.y) {
            const F32 distance = std::min(margin, rect.y - pixel.y);
            delta = -maxStep * std::max(0.1f, distance / std::max(1.0f, margin));
        } else if (pixel.y > rect.bottom()) {
            const F32 distance = std::min(margin, pixel.y - rect.bottom());
            delta = maxStep * std::max(0.1f, distance / std::max(1.0f, margin));
        }
        if (std::abs(delta) > 1e-3f) {
            currentScrollY = std::clamp(currentScrollY + delta, 0.0f,
                                        std::max(0.0f, contentHeightPixels() - rect.height));
        }
    }

    Position positionFromMouse(const Extent2D<F32>& pixel) const {
        ensureVisualRows();
        const F32 contentY = pixel.y - viewportTopPixels() + currentScrollY;
        U64 visualRowIndex = variableMetrics()
            ? visualRowAtContentY(contentY)
            : static_cast<U64>(std::max(0.0f, std::floor(contentY / lineHeightPixels())));
        visualRowIndex = std::min<U64>(visualRowIndex, visualRows.size() - 1);
        const auto& row = visualRows[visualRowIndex];
        const F32 localX = std::max(0.0f, pixel.x - lineTextLeftPixels(row.line)) + currentScrollX;
        const F32 lineX = columnOffsetPixels(row.line, row.start) + localX;
        const U64 column = columnAtOffsetPixels(row.line, lineX);
        Position position;
        position.line = row.line;
        position.column = std::clamp<U64>(column, row.start, row.end);
        return position;
    }

    bool pointerInside(const Extent2D<F32>& position) const {
        return rect.contains(position) &&
               (!clip.has_value() || clip->contains(position));
    }

    bool handleMouse(const MouseEvent& event) {
        const Extent2D<F32> position{event.position.x, event.position.y};
        switch (event.type) {
            case MouseEventType::Click: {
                if (event.button != MouseButton::Left) {
                    return false;
                }
                if (!pointerInside(position)) {
                    clearFocus();
                    return false;
                }
                if (config.onPositionClick && config.onPositionClick(positionFromMouse(position))) {
                    return true;
                }
                focus();
                const bool extend = (ImGui::GetIO().KeyMods & ImGuiMod_Shift) != 0;
                if (ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left)) {
                    selectWordAt(positionFromMouse(position));
                } else {
                    moveCursorTo(positionFromMouse(position), extend);
                    mouseSelecting = true;
                }
                return true;
            }
            case MouseEventType::Move: {
                if (!mouseSelecting) {
                    if (pointerInside(position)) {
                        const bool link = config.isPositionInteractive &&
                                          config.isPositionInteractive(positionFromMouse(position));
                        ImGui::SetMouseCursor(link ? ImGuiMouseCursor_Hand : ImGuiMouseCursor_TextInput);
                    }
                    return false;
                }
                autoscrollSelection(position);
                moveCursorTo(positionFromMouse(position), true);
                return true;
            }
            case MouseEventType::Release: {
                if (event.button != MouseButton::Left || !mouseSelecting) {
                    return false;
                }
                mouseSelecting = false;
                return true;
            }
            default:
                return false;
        }
    }

    void reconcileExternalFocusLoss() {
        if (!hasFocus()) {
            return;
        }
        if ((ImGui::IsMouseClicked(ImGuiMouseButton_Left) || ImGui::IsMouseClicked(ImGuiMouseButton_Right)) &&
            !hovered) {
            clearFocus();
        }
    }

    void handleKeyboard() {
        if (!hasFocus() || !windowFocused) {
            return;
        }
        ImGuiIO& io = ImGui::GetIO();
        const bool commandPressed = (io.KeyMods & ImGuiMod_Super) != 0;
        const bool controlPressed = (io.KeyMods & ImGuiMod_Ctrl) != 0;
        const bool altPressed = (io.KeyMods & ImGuiMod_Alt) != 0;
        const bool shiftPressed = (io.KeyMods & ImGuiMod_Shift) != 0;
        const bool shortcutPressed = commandPressed || controlPressed;
        const bool wordModifier = altPressed || (controlPressed && !commandPressed);

        if (ImGui::IsKeyPressed(ImGuiKey_Escape, false)) {
            if (hasSelection()) {
                clearSelection();
                configureScrollView();
                rebuildVisibleInstances();
            } else {
                clearFocus();
            }
        }

        bool handledShortcut = false;
        if (shortcutPressed) {
            if (ImGui::IsKeyPressed(ImGuiKey_A, false)) {
                selectAll();
                handledShortcut = true;
            }
            if (ImGui::IsKeyPressed(ImGuiKey_C, false)) {
                copySelectionOrLine();
                handledShortcut = true;
            }
            if (config.editable) {
                if (controlPressed && !commandPressed && !altPressed &&
                    (ImGui::IsKeyPressed(ImGuiKey_Enter, false) || ImGui::IsKeyPressed(ImGuiKey_KeypadEnter, false))) {
                    if (config.onSubmit) {
                        config.onSubmit(textValue());
                    }
                    handledShortcut = true;
                }
                if (ImGui::IsKeyPressed(ImGuiKey_X, false)) {
                    cutSelectionOrLine();
                    handledShortcut = true;
                }
                if (ImGui::IsKeyPressed(ImGuiKey_V, false)) {
                    const char* clipboardText = ImGui::GetClipboardText();
                    if (clipboardText != nullptr) {
                        insertText(clipboardText);
                    }
                    handledShortcut = true;
                }
                if (ImGui::IsKeyPressed(ImGuiKey_Z, true)) {
                    if (shiftPressed) {
                        redo();
                    } else {
                        undo();
                    }
                    handledShortcut = true;
                }
                if (ImGui::IsKeyPressed(ImGuiKey_Y, true)) {
                    redo();
                    handledShortcut = true;
                }
            }
        }

        if (!config.editable) {
            return;
        }

        if (!handledShortcut) {
            if (ImGui::IsKeyPressed(ImGuiKey_Backspace, true)) {
                if (commandPressed) {
                    deleteToPosition(lineStartPosition());
                } else if (wordModifier) {
                    deleteToPosition(previousWordPosition());
                } else {
                    backspace();
                }
            }
            if (ImGui::IsKeyPressed(ImGuiKey_Delete, true)) {
                if (commandPressed) {
                    deleteToPosition(lineEndPosition());
                } else if (wordModifier) {
                    deleteToPosition(nextWordPosition());
                } else {
                    deleteForward();
                }
            }
            if (!commandPressed && !controlPressed && !altPressed) {
                if (config.submitOnEnter && !shiftPressed) {
                    if ((ImGui::IsKeyPressed(ImGuiKey_Enter, false) ||
                         ImGui::IsKeyPressed(ImGuiKey_KeypadEnter, false)) &&
                        config.onSubmit) {
                        config.onSubmit(textValue());
                    }
                } else if (ImGui::IsKeyPressed(ImGuiKey_Enter, true) ||
                           ImGui::IsKeyPressed(ImGuiKey_KeypadEnter, true)) {
                    insertNewLine();
                }
            }
            if (!commandPressed && !controlPressed && !altPressed && ImGui::IsKeyPressed(ImGuiKey_Tab, true)) {
                indentSelection(shiftPressed);
            }
            if (ImGui::IsKeyPressed(ImGuiKey_LeftArrow, true)) {
                if (hasSelection() && !shiftPressed) {
                    moveCursorTo(selectionRange().first, false);
                } else {
                    moveCursorTo(commandPressed ? lineStartPosition()
                                                : (wordModifier ? previousWordPosition() : previousCharacterPosition()),
                                 shiftPressed);
                }
            }
            if (ImGui::IsKeyPressed(ImGuiKey_RightArrow, true)) {
                if (altPressed && shiftPressed && !commandPressed && !controlPressed) {
                    expandSyntaxSelection();
                } else if (hasSelection() && !shiftPressed) {
                    moveCursorTo(selectionRange().second, false);
                } else {
                    moveCursorTo(commandPressed ? lineEndPosition()
                                                : (wordModifier ? nextWordPosition() : nextCharacterPosition()),
                                 shiftPressed);
                }
            }
            if (ImGui::IsKeyPressed(ImGuiKey_UpArrow, true)) {
                if (commandPressed) {
                    moveCursorTo(documentStartPosition(), shiftPressed);
                } else {
                    moveCursorVertically(-1, shiftPressed);
                }
            }
            if (ImGui::IsKeyPressed(ImGuiKey_DownArrow, true)) {
                if (commandPressed) {
                    moveCursorTo(documentEndPosition(), shiftPressed);
                } else {
                    moveCursorVertically(1, shiftPressed);
                }
            }
            if (ImGui::IsKeyPressed(ImGuiKey_Home, true)) {
                moveCursorTo(smartLineStartPosition(), shiftPressed);
            }
            if (ImGui::IsKeyPressed(ImGuiKey_End, true)) {
                moveCursorTo(lineEndPosition(), shiftPressed);
            }
            if (ImGui::IsKeyPressed(ImGuiKey_PageUp, true)) {
                moveCursorVertically(-static_cast<I64>(pageStepLines()), shiftPressed);
            }
            if (ImGui::IsKeyPressed(ImGuiKey_PageDown, true)) {
                moveCursorVertically(static_cast<I64>(pageStepLines()), shiftPressed);
            }
            if (!commandPressed && !controlPressed && !altPressed) {
                for (const ImWchar character : io.InputQueueCharacters) {
                    if (character >= 32 && character < 127) {
                        insertTypedCharacter(static_cast<char>(character));
                    }
                }
            }
        }
    }

    void reconcileCursorBlink() {
        if (!config.editable || !hasFocus()) {
            return;
        }
        const bool blinkOn = std::fmod(ImGui::GetTime() - blinkBase, kCursorBlinkPeriod) < kCursorBlinkOnDuration;
        if (blinkOn != lastBlinkOn) {
            rebuildVisibleInstances();
        }
    }

    void clampScroll() {
        currentScrollY = std::clamp(currentScrollY, 0.0f, std::max(0.0f, contentHeightPixels() - rect.height));
        currentScrollX = std::clamp(currentScrollX, 0.0f, std::max(0.0f, maxLineAdvancePixels() - textViewportWidthPixels()));
    }

    void configureScrollView() {
        scroll.update({
            .id = config.id + ":scroll",
            .contentWidth = contentWidthPixels(),
            .contentHeight = contentHeightPixels(),
            .scrollX = currentScrollX,
            .scrollY = currentScrollY,
            .scrollbar = config.scrollbar,
            .wheelStep = lineHeightPixels() * kWheelScrollLines,
            .thickness = fontSizePixels * kScrollbarThicknessFontRatio,
            .margin = fontSizePixels * kScrollbarMarginFontRatio,
            .trackColorKey = config.scrollbarTrackColorKey,
            .thumbColorKey = config.scrollbarThumbColorKey,
            .onScrollY = [this](F32 scrollY) {
                currentScrollY = scrollY;
                rebuildVisibleInstances();
            },
            .onScrollX = [this](F32 scrollX) {
                currentScrollX = scrollX;
                rebuildVisibleInstances();
            },
        });
    }

    Rect clipped(const Rect& r) const {
        if (!clip.has_value()) {
            return r;
        }
        const auto& c = *clip;
        const F32 x = std::max(r.x, c.x);
        const F32 y = std::max(r.y, c.y);
        const F32 right = std::min(r.right(), c.right());
        const F32 bottom = std::min(r.bottom(), c.bottom());
        return {x, y, std::max(0.0f, right - x), std::max(0.0f, bottom - y)};
    }

    void rebuildVisibleInstances() {
        const auto& rect = this->rect;
        const bool visible = !rect.empty();
        const auto clip = std::optional<Rect>(clipped(rect));
        const auto textClip = std::optional<Rect>(clipped(
            Rect{textLeftPixels(), rect.y, std::max(0.0f, textClipRightPixels() - textLeftPixels()), rect.height}));
        const F32 lineHeight = lineHeightPixels();
        const F32 advance = characterAdvancePixels();
        const F32 gutterWidth = gutterWidthPixels();
        const F32 textLeft = textLeftPixels();
        const F32 viewportTop = viewportTopPixels();
        const F32 contentSize = contentFontSize();
        ensureVisualRows();
        const U64 firstVisualRow = firstVisibleVisualRow();
        const auto selection = selectionRange();
        const bool selected = hasSelection();

        static const std::vector<std::vector<StyleId>> kNoStyles;
        const auto& styles = config.styler ? config.styler(lines, contentRevision) : kNoStyles;

        const U64 maxSegments = std::max<U64>(1, config.maxLineSegments);
        const bool hasStyleBackgrounds = !theme.styleBackgrounds.empty();
        std::vector<Label::Instance> codeInstances(kVisibleLineCapacity * maxSegments);
        std::vector<std::vector<Label::Instance>> extraInstances(
            extraFontNames.size(),
            std::vector<Label::Instance>(kVisibleLineCapacity * maxSegments));
        std::vector<Box::Instance> styleBgInstances(
            hasStyleBackgrounds ? kVisibleLineCapacity * maxSegments : 0);
        std::vector<Label::Instance> numberInstances(kVisibleLineCapacity);
        std::vector<Box::Instance> selectionInstances(kVisibleLineCapacity);
        std::vector<Box::Instance> matchInstances(kSelectionMatchCapacity);
        const auto matchText = singleLineSelectionText();
        U64 matchIndex = 0;

        const F32 clipLeft = textLeft;
        const F32 clipRight = textClipRightPixels();
        const auto rowRangeRect = [&](U64 lineIndex, U64 a, U64 b, U64 rowStart, U64 rowEnd, U64 lineLen,
                                      bool breakAtEnd, F32 rowTop, F32 rowHeight) -> std::optional<Rect> {
            const F32 textLeft = lineTextLeftPixels(lineIndex);
            const U64 s = std::max(a, rowStart);
            const U64 e = std::min(b, rowEnd);
            const bool extend = breakAtEnd && rowEnd >= lineLen;
            if (e <= s && !extend) {
                return std::nullopt;
            }
            const F32 startX = textLeft - currentScrollX + columnXInRow(lineIndex, rowStart, s);
            F32 endX = textLeft - currentScrollX + columnXInRow(lineIndex, rowStart, e);
            if (extend) {
                endX = std::max(endX, textLeft - currentScrollX + columnXInRow(lineIndex, rowStart, lineLen) + advance);
            }
            const F32 clippedStart = std::max(startX, clipLeft);
            const F32 clippedEnd = std::min(endX, clipRight);
            if (clippedEnd - clippedStart <= 0.0f) {
                return std::nullopt;
            }
            return Rect{clippedStart, rowTop, clippedEnd - clippedStart, rowHeight};
        };

        static const std::vector<StyleId> kNoLineStyles;
        for (U64 i = 0; i < kVisibleLineCapacity; ++i) {
            const U64 visualRow = firstVisualRow + i;
            const F32 rowTop = viewportTop + rowTopContent(visualRow) - currentScrollY;
            const bool rowVisible = visible && visualRow < visualRows.size() && rowTop < rect.bottom();
            for (U64 segment = 0; segment < maxSegments; ++segment) {
                const U64 slot = i * maxSegments + segment;
                codeInstances[slot].visible = false;
                for (auto& pool : extraInstances) {
                    pool[slot].visible = false;
                }
                if (hasStyleBackgrounds) {
                    styleBgInstances[slot].visible = false;
                }
            }
            if (!rowVisible) {
                numberInstances[i].visible = false;
                selectionInstances[i].visible = false;
                continue;
            }

            const VisualRow row = visualRows[visualRow];
            const auto& line = lines[row.line];
            const U64 lineLen = line.size();
            const auto& lineStyles = row.line < styles.size() ? styles[row.line] : kNoLineStyles;
            const F32 rh = variableMetrics() ? row.height : lineHeight;
            const F32 lineLeft = lineTextLeftPixels(row.line);
            const F32 lineSize = lineGlyphSize(row.line);

            U64 startColumn = row.start;
            U64 segmentIndex = 0;
            while (startColumn < row.end && segmentIndex < maxSegments) {
                const StyleId style = startColumn < lineStyles.size() ? lineStyles[startColumn] : 0;
                U64 endColumn = startColumn + 1;
                while (endColumn < row.end &&
                       (endColumn < lineStyles.size() ? lineStyles[endColumn] : 0) == style &&
                       endColumn - startColumn < kTextSegmentCharacterCapacity) {
                    ++endColumn;
                }
                const U64 slot = i * maxSegments + segmentIndex;
                const F32 segX = lineLeft - currentScrollX + columnXInRow(row.line, row.start, startColumn);
                const F32 segW = columnOffsetPixels(row.line, endColumn) - columnOffsetPixels(row.line, startColumn);
                const U64 poolIndex = poolIndexForFont(fontForStyle(style));
                auto& target = poolIndex == static_cast<U64>(-1) ? codeInstances : extraInstances[poolIndex];
                target[slot] = {
                    .rect = {segX, rowTop, segW, rh},
                    .str = line.substr(startColumn, endColumn - startColumn),
                    .visible = true,
                    .color = colorForStyle(style),
                    .fontSize = lineSize,
                    .alignment = {0, 1},
                };
                if (hasStyleBackgrounds) {
                    const auto bg = backgroundForStyle(style);
                    if (bg.a > 0.0f) {
                        const F32 padX = rh * 0.12f;
                        styleBgInstances[slot] = {
                            .rect = {segX - padX, rowTop + rh * 0.08f,
                                     segW + 2.0f * padX, rh * 0.84f},
                            .visible = true,
                            .backgroundColor = bg,
                        };
                    }
                }
                ++segmentIndex;
                startColumn = endColumn;
            }

            numberInstances[i] = {
                .rect = {rect.x, rowTop, lineNumberRightPixels(), rh},
                .str = jst::fmt::format("{}", row.line + 1),
                .visible = config.lineNumbers && row.start == 0,
                .color = theme.lineNumber,
                .fontSize = contentSize,
                .alignment = {2, 1},
            };

            selectionInstances[i].visible = false;
            if (selected && row.line >= selection.first.line && row.line <= selection.second.line) {
                const U64 lo = row.line == selection.first.line ? selection.first.column : 0;
                const U64 hi = row.line == selection.second.line ? selection.second.column : lineLen;
                const bool breakAtEnd = row.line < selection.second.line;
                if (const auto r = rowRangeRect(row.line, lo, hi, row.start, row.end, lineLen, breakAtEnd, rowTop, rh)) {
                    selectionInstances[i] = {.rect = *r, .backgroundColor = theme.selection};
                }
            }

            if (matchText.has_value()) {
                std::size_t found = line.find(*matchText);
                while (found != std::string::npos && matchIndex < kSelectionMatchCapacity) {
                    const U64 startCol = static_cast<U64>(found);
                    const U64 endCol = startCol + matchText->size();
                    const bool isPrimary = selected && row.line == selection.first.line &&
                                           startCol == selection.first.column && endCol == selection.second.column;
                    if (!isPrimary) {
                        if (const auto r = rowRangeRect(row.line, startCol, endCol, row.start, row.end, lineLen, false, rowTop, rh)) {
                            matchInstances[matchIndex++] = {.rect = *r, .backgroundColor = theme.selectionMatch};
                        }
                    }
                    found = line.find(*matchText, found + matchText->size());
                }
            }
        }

        backgroundBox.update({
            .id = config.id + ":background",
            .instances = {{.rect = rect, .visible = visible, .backgroundColor = theme.background}},
        });

        const U64 cursorVisualRow = visualRowForPosition(cursor);
        const F32 activeRowTop = rowTopPixels(cursorVisualRow);
        activeLineBox.update({
            .id = config.id + ":active-line",
            .instances = {{
                .rect = {rect.x, activeRowTop, rect.width, lineHeight},
                .visible = visible && config.showActiveLine && hasFocus() && !selected &&
                            activeRowTop + lineHeight > rect.y && activeRowTop < rect.bottom(),
                .backgroundColor = theme.activeLine,
            }},
            .clip = clip,
        });

        selectionMatchBox.update({
            .id = config.id + ":selection-match",
            .instances = std::move(matchInstances),
            .clip = textClip,
            .capacity = kSelectionMatchCapacity,
        });
        selectionBox.update({
            .id = config.id + ":selection",
            .instances = std::move(selectionInstances),
            .clip = textClip,
            .capacity = kVisibleLineCapacity,
        });
        styleBackgroundBox.update({
            .id = config.id + ":style-bg",
            .instances = std::move(styleBgInstances),
            .clip = textClip,
            .cornerRadius = lineHeightPixels() * 0.22f,
            .capacity = hasStyleBackgrounds ? kVisibleLineCapacity * maxSegments : 0,
        });
        codeLabels.update({
            .id = config.id + ":text",
            .instances = std::move(codeInstances),
            .clip = textClip,
            .fontName = config.fontName,
            .maxCharacters = kTextSegmentCharacterCapacity,
            .capacity = kVisibleLineCapacity * maxSegments,
        });
        for (U64 k = 0; k < extraFontLabels.size(); ++k) {
            if (k < extraFontNames.size()) {
                extraFontLabels[k]->update({
                    .id = config.id + ":text-font" + std::to_string(k),
                    .instances = std::move(extraInstances[k]),
                    .clip = textClip,
                    .fontName = extraFontNames[k],
                    .maxCharacters = kTextSegmentCharacterCapacity,
                    .capacity = kVisibleLineCapacity * maxSegments,
                });
            } else {
                extraFontLabels[k]->update({
                    .id = config.id + ":text-font" + std::to_string(k),
                    .instances = {},
                    .clip = textClip,
                    .fontName = config.fontName,
                    .maxCharacters = kTextSegmentCharacterCapacity,
                    .capacity = kVisibleLineCapacity * maxSegments,
                });
            }
        }
        numberLabels.update({
            .id = config.id + ":numbers",
            .instances = std::move(numberInstances),
            .clip = clip,
            .maxCharacters = kMaxLineNumberCharacters,
            .capacity = kVisibleLineCapacity,
        });

        const F32 separatorWidth = std::max(1.0f, std::round(fontSizePixels * kSeparatorWidthFontRatio));
        const F32 separatorCenter = rect.x + gutterWidth - kLineNumberRightPaddingCharacters * 0.5f * advance;
        gutterBox.update({
            .id = config.id + ":gutter",
            .instances = {{
                .rect = {separatorCenter - separatorWidth * 0.5f, rect.y, separatorWidth, rect.height},
                .visible = visible && config.lineNumbers,
                .backgroundColor = theme.gutterSeparator,
            }},
            .clip = clip,
        });

        const bool blinkOn = std::fmod(ImGui::GetTime() - blinkBase, kCursorBlinkPeriod) < kCursorBlinkOnDuration;
        lastBlinkOn = blinkOn;
        const F32 cursorWidth = std::max(1.0f, std::round(fontSizePixels * kCursorWidthFontRatio));
        const F32 cursorTop = rowTopPixels(cursorVisualRow);
        const U64 cursorRowStart = cursorVisualRow < visualRows.size() ? visualRows[cursorVisualRow].start : 0;
        const F32 cursorX = textLeft - currentScrollX +
                            columnXInRow(cursor.line, cursorRowStart, cursor.column);
        cursorBox.update({
            .id = config.id + ":cursor",
            .instances = {{
                .rect = {cursorX, cursorTop, cursorWidth, lineHeight},
                .visible = visible && config.editable && hasFocus() && blinkOn &&
                            cursorTop + lineHeight > rect.y && cursorTop < rect.bottom(),
                .backgroundColor = theme.cursor,
            }},
            .clip = textClip,
        });
    }

};

TextGrid::TextGrid() {
    this->impl = std::make_unique<Impl>();
    setClipsChildren(true);
    this->impl->addChild = [this](Component& child) {
        add(child);
    };
    add(this->impl->backgroundBox);
    add(this->impl->activeLineBox);
    add(this->impl->selectionMatchBox);
    add(this->impl->selectionBox);
    add(this->impl->styleBackgroundBox);
    add(this->impl->gutterBox);
    add(this->impl->numberLabels);
    add(this->impl->codeLabels);
    add(this->impl->cursorBox);
    add(this->impl->scroll);
}

TextGrid::~TextGrid() {
    if (impl && impl->focused && ActiveTextGridId() == impl->config.id) {
        ActiveTextGridId().clear();
        Private::SetKeyboardInputCaptured(false);
    }
}

bool TextGrid::update(Config config) {

    const bool valueChanged = impl->config.value != config.value;
    const bool wasAtBottom = impl->rect.height <= 0.0f ||
                             impl->currentScrollY + 1.0f >=
                                 std::max(0.0f, impl->contentHeightPixels() - impl->rect.height);
    impl->config = std::move(config);
    impl->fontSizePixels = impl->config.fontSize;

    if ((valueChanged && impl->config.value != impl->textValue()) || impl->lines.empty()) {
        impl->lines = SplitLines(impl->config.value);
        ++impl->contentRevision;
        if (impl->config.editable) {
            impl->undoStack.clear();
            impl->redoStack.clear();
            impl->cursor = {0, 0};
            impl->selectionAnchor = {0, 0};
            impl->selectionActive = false;
            impl->preferredColumn.reset();
            impl->currentScrollY = 0.0f;
            impl->currentScrollX = 0.0f;
            impl->stickToBottomPending = false;
        } else {
            impl->clampPosition(impl->cursor);
            impl->clampPosition(impl->selectionAnchor);
            impl->stickToBottomPending = impl->config.stickToBottom && wasAtBottom;
        }
    }

    return true;
}

const TextGrid::Metrics& TextGrid::metrics() const {
    return this->impl->storedMetrics;
}

Extent2D<F32> TextGrid::measure(const Context& ctx, Extent2D<F32> available) {
    impl->textMetrics.setWindow(ctx.render);

    const F32 maxWidth = std::isfinite(available.x) ? available.x : impl->rect.width;
    const Rect savedRect = impl->rect;
    impl->rect = {0.0f, 0.0f, maxWidth, impl->rect.height};
    const F32 height = impl->textContentHeightPixels();
    const F32 width = std::min(maxWidth, impl->measuredContentWidthPixels());
    impl->rect = savedRect;

    return {width, height};
}

void TextGrid::layout(const Context& ctx) {

    impl->rect = frame();
    impl->clip = std::optional<Rect>(Intersect(frame(), clip()));
    impl->hovered = ctx.hovered;
    impl->active = ctx.active;
    impl->windowFocused = ctx.windowFocused;

    impl->textMetrics.setWindow(ctx.render);
    impl->resolveTheme(ctx);
    impl->ensureFontPools();
    if (impl->stickToBottomPending) {
        impl->currentScrollY = std::max(0.0f, impl->contentHeightPixels() - impl->rect.height);
        impl->stickToBottomPending = false;
    }
    impl->clampScroll();
    impl->configureScrollView();
    layoutChild(ctx, impl->scroll, frame());
    impl->rebuildVisibleInstances();

    impl->reconcileFocusOwnership();
    impl->reconcileExternalFocusLoss();
    impl->handleKeyboard();
    impl->reconcileCursorBlink();
    impl->notifyLayout();

}

bool TextGrid::event(const MouseEvent& event) {
    if (eventChildren(event)) {
        return true;
    }
    return impl->handleMouse(event);
}

}  // namespace Jetstream::Sakura::Retained
