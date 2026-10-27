#include <jetstream/render/sakura/components/text_grid.hh>

#include <jetstream/render/components/text.hh>
#include <jetstream/render/sakura/clipboard.hh>
#include <jetstream/render/sakura/components/box.hh>
#include <jetstream/render/sakura/components/label.hh>
#include <jetstream/render/sakura/components/scroller.hh>

#include "../context.hh"
#include "../metrics.hh"
#include "../detail/helpers.hh"
#include "../detail/text_grid_viewport.hh"
#include "../detail/text_metrics.hh"
#include "tools/text.hh"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace Jetstream::Sakura {

using Jetstream::Text::SplitLines;
using Jetstream::Text::JoinLines;

namespace {

using Unicode = Jetstream::Render::Components::Text::Unicode;

constexpr F32 kStyleBackgroundHeightRatio = 1.3f;
constexpr F32 kStyleBackgroundCornerRatio = 0.25f;
constexpr F32 kScrollbarThicknessFontRatio = 6.0f / Typography::FontSize;
constexpr F32 kScrollbarMarginFontRatio = 4.0f / Typography::FontSize;
constexpr F32 kSeparatorWidthFontRatio = 1.0f / Typography::FontSize;
constexpr F32 kCursorWidthFontRatio = 2.0f / Typography::FontSize;
constexpr F32 kCursorScrollMarginFontRatio = 24.0f / Typography::FontSize;
constexpr F32 kLineNumberLeftPaddingCharacters = 1.0f;
constexpr F32 kLineNumberRightPaddingCharacters = 1.5f;
constexpr U64 kLineNumberDigits = 4;
constexpr U64 kMaxLineNumberCharacters = 6;
constexpr F32 kWheelScrollLines = 3.0f;
constexpr F32 kDragScrollMarginFontRatio = 36.0f / Typography::FontSize;
constexpr F32 kDragScrollMaxLines = 0.85f;
constexpr F32 kFallbackAdvanceFontRatio = 0.5f;
constexpr F32 kWrapTrailingMarginCharacters = 0.5f;
constexpr F32 kScrollRangeEpsilon = 0.5f;
constexpr U64 kTextSegmentCharacterCapacity = 32;
constexpr U64 kMinimumLineScaleDivisor = 2;
constexpr F32 kMinimumLineScale = 1.0f / static_cast<F32>(kMinimumLineScaleDivisor);
constexpr U64 kSegmentByteFloor = 128;
constexpr U64 kMaxCodepointBytes = 4;
constexpr U64 kSegmentLimitScale = (kSegmentByteFloor + kTextSegmentCharacterCapacity - kMaxCodepointBytes) /
                                   (kTextSegmentCharacterCapacity - kMaxCodepointBytes + 1);
constexpr U64 kSelectionMatchCapacity = 128;
constexpr U64 kMaxUndoHistory = 128;
constexpr U64 kTabSize = 4;
constexpr F64 kCursorBlinkPeriod = 1.0;
constexpr F64 kCursorBlinkOnDuration = 0.55;
constexpr F32 kHitContainsWeight = 16777216.0f;
constexpr F32 kHitVerticalWeight = 4096.0f;
constexpr std::string_view kAutoPairOpenings = "([{\"'";
constexpr std::string_view kAutoPairClosings = ")]}\"'";

bool IsWordChar(char character) {
    return std::isalnum(static_cast<unsigned char>(character)) || character == '_' ||
           static_cast<unsigned char>(character) >= 0x80;
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

std::string MaskedRange(const std::string& line, U64 start, U64 end) {
    std::string masked;
    for (U64 pos = start; pos < end; pos = Unicode::NextCharacter(line, pos)) {
        masked += '*';
    }
    return masked;
}

U64 LeadingWhitespaceColumn(const std::string& line) {
    U64 column = 0;
    while (column < line.size() && (line[column] == ' ' || line[column] == '\t')) {
        ++column;
    }
    return column;
}

U64 LastBreakBefore(std::string_view text, U64 from, U64 limit) {
    for (U64 i = limit; i > from; --i) {
        if (text[i - 1] == ' ' || text[i - 1] == '\t') {
            return i;
        }
    }
    return from;
}

std::vector<std::pair<U64, U64>> WrapSegments(std::string_view line, U64 cols, bool word) {
    std::vector<std::pair<U64, U64>> segments;
    const U64 len = line.size();
    if (cols == 0 || len <= cols) {
        segments.emplace_back(0, len);
        return segments;
    }
    U64 pos = 0;
    while (pos < len) {
        const U64 limit = Unicode::Advance(line, pos, cols);
        U64 end = limit;
        if (word && limit < len) {
            const U64 brk = LastBreakBefore(line, pos, limit);
            end = brk > pos ? brk : limit;
        }
        segments.emplace_back(pos, end);
        pos = end;
    }
    return segments;
}

}  // namespace

struct TextGrid::Impl {
    using Position = TextGrid::Position;

    static constexpr U64 kNoRegion = static_cast<U64>(-1);

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
        std::vector<ColorRGBA<F32>> styleColors;
        std::vector<ColorRGBA<F32>> styleBackgrounds;
    };

    struct VisualRow {
        U64 line = 0;
        U64 start = 0;
        U64 end = 0;
        F32 top = 0.0f;
        F32 height = 0.0f;
    };

    struct Lane {
        U64 group = 0;
        std::string font;
        bool used = false;
        std::unique_ptr<Label> label;
    };

    struct FrameGeometry {
        Rect rect;
        std::optional<Rect> clip;
        std::optional<Rect> textClip;
        std::vector<Rect> groupClips;
        bool visible = false;
        F32 clipTop = 0.0f;
        F32 clipBottom = 0.0f;
        F32 clipLeft = 0.0f;
        F32 clipRight = 0.0f;
        U64 maxSegments = 1;
        U64 lineCapacity = 1;
        U64 segmentCapacity = 1;
    };

    struct InstancePools {
        std::vector<Label::Instance> text;
        std::vector<std::vector<Label::Instance>> extraText;
        std::vector<std::vector<Label::Instance>> laneText;
        std::vector<Box::Instance> styleBackgrounds;
        std::vector<Label::Instance> lineNumbers;
        std::vector<Box::Instance> selection;
        std::vector<Box::Instance> matches;
    };

    struct SelectionState {
        bool active = false;
        Position first;
        Position second;
        std::optional<std::string> matchText;
    };

    struct RowContext {
        const VisualRow& row;
        F32 top;
        F32 height;
        const std::string& line;
        const std::vector<StyleId>& styles;
        F32 glyphSize;
        U64 group;
        F32 scrollX;
        const Rect* clip;
    };

    Config config;
    ResolvedTheme theme;

    Scroller scroll;
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
    std::vector<Lane> lanes;
    std::vector<U64> regionGroup;
    std::vector<U64> groupRegion;
    std::vector<F32> regionScroll;
    std::function<void(Component&)> addChild;
    mutable TextMetrics textMetrics;

    Rect rect;
    std::optional<Rect> clip;
    F32 surfaceHeight = 0.0f;
    Metrics storedMetrics;
    TextGridViewport viewport;

    std::vector<std::string> lines = {""};
    F32 currentScrollX = 0.0f;
    F32 currentScrollY = 0.0f;
    Extent2D<F32> elastic = {0.0f, 0.0f};
    F32 pixelRatio = 1.0f;
    bool stickToBottomPending = false;
    F32 fontSizePixels = Typography::FontSize;

    bool focused = false;
    bool mouseSelecting = false;
    Position cursor;
    Position selectionAnchor;
    bool selectionActive = false;
    std::optional<F32> preferredContentX;
    std::vector<Snapshot> undoStack;
    std::vector<Snapshot> redoStack;
    U64 contentRevision = 0;
    U64 measureGeneration = 0;
    std::vector<U8> resolvedFontAvailability;

    struct MeasureMemo {
        bool valid = false;
        F32 availableWidth = 0.0f;
        F32 height = 0.0f;
        U64 revision = 0;
        U64 generation = 0;
        const Render::Window* window = nullptr;
        Extent2D<F32> result = {0.0f, 0.0f};
        F32 naturalWidth = 0.0f;
    };
    MeasureMemo measureMemo;
    F64 blinkBase = 0.0;
    bool blinkPending = false;
    bool layingOut = false;
    F64 now = 0.0;
    bool lastBlinkOn = false;
    bool windowFocused = true;

    mutable std::vector<VisualRow> visualRows;
    mutable std::vector<U64> groupStartRows;
    mutable std::vector<U64> visualRowStartIndex;
    mutable F32 visualContentHeight = 0.0f;
    mutable bool visualRowsValid = false;
    mutable U64 visualRowsRevision = 0;
    mutable Wrap visualRowsWrap = Wrap::None;
    mutable F32 visualRowsWrapWidth = -1.0f;
    mutable F32 visualRowsAdvance = -1.0f;
    std::vector<std::pair<U64, U64>> visibleRowRanges;

    mutable std::vector<U8> lineAsciiCache;
    mutable U64 lineAsciiRevision = static_cast<U64>(-1);

    mutable std::vector<std::vector<F32>> linePrefixes;
    mutable bool linePrefixesValid = false;
    mutable U64 linePrefixesRevision = static_cast<U64>(-1);
    mutable F32 linePrefixesFontSize = -1.0f;
    mutable std::vector<std::pair<F32, F32>> cellAdvanceCache;
    WidthLayout resolvedWidthLayout;
    bool widthLayoutResolved = false;
    bool gutterReserved = false;
    bool bottomReserved = false;
    bool liveLayout = false;
    bool cursorFollowPending = false;
    TextGrid* owner = nullptr;
    bool offered = false;
    bool selectAllOnOffer = false;
    bool viewportResolved = false;
    bool resolvedLayoutPass = false;
    U64 resolvedRevision = 0;
    Extent2D<F32> resolvedSize = {-1.0f, -1.0f};
    F32 lastNaturalWidth = 0.0f;

    F32 contentFontSize() const { return fontSizePixels * config.fontScale; }
    F32 lineHeightPixels() const { return std::max(1.0f, contentFontSize() * config.lineHeight); }
    F32 lineScaleAt(U64 line) const {
        return line < config.lineScale.size() ? std::max(kMinimumLineScale, config.lineScale[line]) : 1.0f;
    }
    F32 lineTopGapAt(U64 line) const {
        return line < config.lineTopGap.size() ? config.lineTopGap[line] : 0.0f;
    }
    const std::vector<F32>& lineIndents() const {
        return config.widthLayout ? resolvedWidthLayout.lineIndent : config.lineIndent;
    }
    const std::vector<F32>& lineWrapWidths() const {
        return config.widthLayout ? resolvedWidthLayout.lineWrapWidth : config.lineWrapWidth;
    }
    F32 contentMinWidthPixels() const {
        return config.widthLayout ? resolvedWidthLayout.contentMinWidth : config.contentMinWidth;
    }
    F32 lineIndentAt(U64 line) const {
        const auto& indents = lineIndents();
        return line < indents.size() ? indents[line] : 0.0f;
    }
    RowMode lineRowModeAt(U64 line) const {
        const auto& modes = config.widthLayout ? resolvedWidthLayout.lineRowMode : config.lineRowMode;
        return line < modes.size() ? modes[line] : RowMode::Next;
    }
    bool lineContinuesRowAt(U64 line) const {
        return lineRowModeAt(line) != RowMode::Next;
    }
    F32 lineWrapWidthAt(U64 line) const {
        const auto& widths = lineWrapWidths();
        return line < widths.size() ? widths[line] : 0.0f;
    }
    F32 lineRightInsetAt(U64 line) const {
        return line < config.lineRightInset.size() ? config.lineRightInset[line] : 0.0f;
    }
    F32 lineTrailingMarginPixels(U64 line) const {
        if (config.wrap == Wrap::None || lineWrapWidthAt(line) > 0.0f || lineWrapsByColumns(line)) {
            return 0.0f;
        }
        return kWrapTrailingMarginCharacters * lineAdvancePixels(line);
    }
    F32 lineGlyphSize(U64 line) const { return contentFontSize() * lineScaleAt(line); }
    F32 lineHeightAt(U64 line) const {
        return std::max(1.0f, lineGlyphSize(line) * config.lineHeight);
    }
    F32 cellAdvancePixels(F32 glyphSize) const {
        for (const auto& [size, advance] : cellAdvanceCache) {
            if (size == glyphSize) {
                return advance;
            }
        }
        const F32 measured = textMetrics.measure(config.fontName, "0", glyphSize);
        const F32 advance = measured > 0.0f ? measured : glyphSize * kFallbackAdvanceFontRatio;
        cellAdvanceCache.emplace_back(glyphSize, advance);
        return advance;
    }
    void beginMetricsFrame(Render::Window* window) {
        textMetrics.setWindow(window);
        cellAdvanceCache.clear();
        auto fonts = FontAvailability(textMetrics.window, config.fontName, extraFontNames);
        if (fonts != resolvedFontAvailability) {
            resolvedFontAvailability = std::move(fonts);
            linePrefixesValid = false;
            visualRowsValid = false;
            viewportResolved = false;
            ++measureGeneration;
        }
    }
    F32 lineAdvancePixels(U64 line) const { return cellAdvancePixels(lineGlyphSize(line)); }
    U64 visibleLineCapacity() const {
        return std::max<U64>(1, config.visibleLineCapacity);
    }

    Padding paddingPixels() const {
        if (config.padding.has_value()) {
            return *config.padding;
        }
        const F32 horizontal = fontSizePixels * Typography::TextPaddingRatio;
        return {horizontal, 0.0f, horizontal, 0.0f};
    }
    F32 gutterWidthPixels() const {
        if (!config.lineNumbers) {
            return 0.0f;
        }
        return (kLineNumberLeftPaddingCharacters + static_cast<F32>(kLineNumberDigits) +
                kLineNumberRightPaddingCharacters) * cellAdvancePixels(contentFontSize());
    }
    F32 lineNumberRightPixels() const {
        return (kLineNumberLeftPaddingCharacters + static_cast<F32>(kLineNumberDigits)) *
               cellAdvancePixels(contentFontSize());
    }
    F32 textLeftPixels() const {
        return config.lineNumbers ? rect.x + gutterWidthPixels() : rect.x + paddingPixels().left;
    }
    F32 textTopPixels() const { return rect.y + paddingPixels().top; }
    F32 textClipLeftPixels() const { return config.lineNumbers ? textLeftPixels() : rect.x; }
    F32 textClipRightPixels() const { return rect.right(); }
    F32 scrollbarReservePixels() const {
        return fontSizePixels * (kScrollbarThicknessFontRatio + 2.0f * kScrollbarMarginFontRatio) + config.scrollbarInset.x;
    }
    F32 scrollbarGutterPixels() const {
        return config.scrollbar && gutterReserved ? scrollbarReservePixels() : 0.0f;
    }
    F32 scrollbarBottomPixels() const {
        return config.scrollbar && bottomReserved ? scrollbarReservePixels() : 0.0f;
    }
    F32 textViewportWidthPixels() const {
        return std::max(1.0f, rect.right() - scrollbarGutterPixels() - paddingPixels().right - textLeftPixels());
    }

    F32 screenXFromContent(F32 contentX) const { return textLeftPixels() - currentScrollX - elastic.x + contentX; }
    F32 contentXFromScreen(F32 screenX) const { return screenX - textLeftPixels() + currentScrollX + elastic.x; }
    F32 screenYFromContent(F32 contentY) const { return textTopPixels() - currentScrollY - elastic.y + contentY; }
    F32 contentYFromScreen(F32 screenY) const { return screenY - textTopPixels() + currentScrollY + elastic.y; }

    F32 rowContentLeft(const VisualRow& row) const { return lineIndentAt(row.line); }
    F32 rowContentWidth(const VisualRow& row) const { return columnXInRow(row.line, row.start, row.end); }
    F32 contentXAtColumn(const VisualRow& row, U64 column) const {
        return rowContentLeft(row) + columnXInRow(row.line, row.start, column);
    }
    U64 columnAtContentX(const VisualRow& row, F32 contentX) const {
        const F32 lineX = columnOffsetPixels(row.line, row.start) + std::max(0.0f, contentX - rowContentLeft(row));
        return std::clamp<U64>(columnAtOffsetPixels(row.line, lineX), row.start, row.end);
    }
    U64 regionOf(U64 line) const {
        const auto& regions = config.scrollRegions;
        const auto it = std::upper_bound(regions.begin(), regions.end(), line,
                                         [](U64 value, const ScrollRegion& region) {
                                             return value < region.first;
                                         });
        if (it == regions.begin() || line > (it - 1)->last) {
            return kNoRegion;
        }
        return static_cast<U64>(it - 1 - regions.begin());
    }
    F32 lineScrollX(U64 line) const {
        const U64 region = regionOf(line);
        return region < regionScroll.size() ? regionScroll[region] : 0.0f;
    }
    F32 regionLeft(U64 region) const { return lineIndentAt(config.scrollRegions[region].first); }
    F32 regionRight(U64 region) const {
        const U64 first = config.scrollRegions[region].first;
        return std::max(regionLeft(region) + lineAdvancePixels(first),
                        textViewportWidthPixels() - lineRightInsetAt(first));
    }
    F32 regionMaxScrollX(U64 region) const {
        const auto& span = config.scrollRegions[region];
        F32 widest = 0.0f;
        for (U64 line = span.first; line <= span.last && line < lines.size(); ++line) {
            widest = std::max(widest, lineIndentAt(line) + lineWidthPixels(line));
        }
        const F32 range = widest - regionRight(region);
        return range > kScrollRangeEpsilon ? range : 0.0f;
    }

    F32 contentDistanceX(const VisualRow& row, F32 contentX) const {
        const F32 left = rowContentLeft(row);
        const F32 right = left + rowContentWidth(row);
        return contentX < left ? left - contentX : contentX > right ? contentX - right : 0.0f;
    }

    bool lineIsAscii(U64 lineIndex) const {
        if (lineAsciiRevision != contentRevision || lineAsciiCache.size() != lines.size()) {
            lineAsciiCache.resize(lines.size());
            for (U64 i = 0; i < lines.size(); ++i) {
                lineAsciiCache[i] = Unicode::IsAscii(lines[i]) ? 1 : 0;
            }
            lineAsciiRevision = contentRevision;
        }
        return lineAsciiCache[lineIndex] != 0;
    }
    bool lineUsesCellGeometry(U64 lineIndex) const {
        return config.monospace && lineIsAscii(lineIndex);
    }
    bool lineWrapsByColumns(U64 lineIndex) const {
        return config.monospace && !lineHasIcons(lines[lineIndex]);
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

    void ensureLinePrefixes() const {
        const F32 fontSize = contentFontSize();
        if (linePrefixesValid && linePrefixesRevision == contentRevision &&
            linePrefixesFontSize == fontSize) {
            return;
        }
        const bool ready = fontsReady();

        static const std::vector<std::vector<StyleId>> kNoStyles;
        static const std::vector<StyleId> kNoLineStyles;
        const bool multiFont = styleMetricsActive();
        const auto& styles = multiFont ? config.styler(lines, contentRevision) : kNoStyles;

        linePrefixes.assign(lines.size(), {});
        for (U64 i = 0; i < lines.size(); ++i) {
            const auto& line = lines[i];
            auto& prefix = linePrefixes[i];
            prefix.resize(line.size() + 1, 0.0f);
            const F32 lineSize = lineGlyphSize(i);
            if (!multiFont && !lineHasIcons(line)) {
                const auto adv = textMetrics.advances(config.fontName, line, lineSize);
                for (U64 c = 0; c < line.size(); ++c) {
                    prefix[c + 1] = prefix[c] + (c < adv.size() ? adv[c] : 0.0f);
                }
                continue;
            }
            const auto& lineStyles = i < styles.size() ? styles[i] : kNoLineStyles;
            U64 c = 0;
            while (c < line.size()) {
                const StyleId style = c < lineStyles.size() ? lineStyles[c] : 0;
                bool icon = false;
                const U64 e = runEnd(line, lineStyles, c, line.size(), icon);
                const std::string font = fontForRun(style, icon);
                const F32 glyphSize = lineSize * scaleForStyle(style);
                const auto adv = textMetrics.advances(font, line.substr(c, e - c), glyphSize);
                const F32 pad = stylePaddingPixels(style, glyphSize);
                const bool spanStart = c == 0 || (c - 1 < lineStyles.size() ? lineStyles[c - 1] : 0) != style;
                const bool spanEnd = e >= line.size() || (e < lineStyles.size() ? lineStyles[e] : 0) != style;
                for (U64 k = c; k < e; ++k) {
                    prefix[k + 1] = prefix[k] + (k - c < adv.size() ? adv[k - c] : 0.0f) +
                                    (k == c && spanStart ? pad : 0.0f) + (k + 1 == e && spanEnd ? pad : 0.0f);
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
        if (lineUsesCellGeometry(lineIndex)) {
            return static_cast<F32>(col) * lineAdvancePixels(lineIndex);
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
        if (lineUsesCellGeometry(lineIndex)) {
            const F32 advance = std::max(1.0f, lineAdvancePixels(lineIndex));
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

    F32 lineWrapWidthPixels(U64 line, F32 trailingMargin) const {
        const F32 overrideWidth = lineWrapWidthAt(line);
        if (overrideWidth > 0.0f) {
            return overrideWidth;
        }
        return std::max(1.0f, textViewportWidthPixels() - lineIndentAt(line) - lineRightInsetAt(line) - trailingMargin);
    }

    std::vector<std::pair<U64, U64>> wrapSegmentsByColumns(U64 line, bool word) const {
        const F32 width = lineWrapWidthPixels(line, 0.0f);
        if (!std::isfinite(width)) {
            return {{0, lines[line].size()}};
        }
        const U64 cols = std::max<U64>(1, static_cast<U64>(std::floor(width / std::max(1.0f, lineAdvancePixels(line)))));
        return WrapSegments(lines[line], cols, word);
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
                const U64 brk = LastBreakBefore(text, pos, fitEnd);
                end = brk > pos ? brk : fitEnd;
            }
            if (end < len) {
                const U64 aligned = Unicode::Align(text, end);
                if (aligned != end) {
                    end = Unicode::PreviousCharacter(text, aligned);
                }
                if (end <= pos) {
                    end = Unicode::NextCharacter(text, pos);
                }
            }
            segments.emplace_back(pos, end);
            pos = end;
        }
        if (segments.empty()) {
            segments.emplace_back(0, len);
        }
        return segments;
    }

    std::vector<std::pair<U64, U64>> wrapSegmentsFor(U64 line) const {
        const U64 len = lines[line].size();
        if (config.wrap == Wrap::None || regionOf(line) != kNoRegion) {
            return {{0, len}};
        }
        const bool word = config.wrap == Wrap::Word;
        if (lineWrapsByColumns(line)) {
            return wrapSegmentsByColumns(line, word);
        }
        return wrapSegmentsProportional(line, lineWrapWidthPixels(line, lineTrailingMarginPixels(line)), word);
    }

    void ensureVisualRows() const {
        const F32 wrapWidth = config.wrap == Wrap::None ? 0.0f : textViewportWidthPixels();
        const F32 advance = cellAdvancePixels(contentFontSize());
        if (visualRowsValid && visualRowsRevision == contentRevision &&
            visualRowsWrap == config.wrap && visualRowsWrapWidth == wrapWidth &&
            visualRowsAdvance == advance) {
            return;
        }
        visualRows.clear();
        groupStartRows.clear();
        visualRowStartIndex.assign(lines.size(), 0);
        F32 y = 0.0f;
        F32 maxY = 0.0f;
        F32 rowGroupTop = 0.0f;
        for (U64 line = 0; line < lines.size(); ++line) {
            const RowMode mode = line > 0 ? lineRowModeAt(line) : RowMode::Next;
            if (mode == RowMode::Below) {
                y += lineTopGapAt(line);
            } else if (mode == RowMode::Same) {
                y = rowGroupTop;
            } else {
                y = maxY + lineTopGapAt(line);
                rowGroupTop = y;
                groupStartRows.push_back(visualRows.size());
            }
            visualRowStartIndex[line] = visualRows.size();
            const F32 h = lineHeightAt(line);
            for (const auto& [start, end] : wrapSegmentsFor(line)) {
                visualRows.push_back({line, start, end, y, h});
                y += h;
            }
            maxY = std::max(maxY, y);
        }
        if (visualRows.empty()) {
            visualRows.push_back({0, 0, 0, 0.0f, lineHeightAt(0)});
            maxY = lineHeightAt(0);
        }
        if (groupStartRows.empty()) {
            groupStartRows.push_back(0);
        }
        visualContentHeight = maxY;
        if (config.monospace || config.wrap == Wrap::None || fontsReady()) {
            visualRowsValid = true;
            visualRowsRevision = contentRevision;
            visualRowsWrap = config.wrap;
            visualRowsWrapWidth = wrapWidth;
            visualRowsAdvance = advance;
        }
    }

    F32 rowTopContent(U64 visualRow) const {
        return visualRow < visualRows.size() ? visualRows[visualRow].top : visualContentHeight;
    }
    F32 rowHeightPixels(U64 visualRow) const {
        return visualRow < visualRows.size() ? visualRows[visualRow].height : lineHeightPixels();
    }
    F32 rowTopPixels(U64 visualRow) const {
        return screenYFromContent(rowTopContent(visualRow));
    }

    U64 rowLastColumn(const VisualRow& row) const {
        const bool wrapped = row.line < lines.size() && row.end < lines[row.line].size();
        return wrapped && row.end > row.start ? Unicode::PreviousCharacter(lines[row.line], row.end) : row.end;
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

    U64 visualRowAtContentY(F32 y) const {
        if (visualRows.empty() || groupStartRows.empty()) {
            return 0;
        }
        U64 lo = 0;
        U64 hi = groupStartRows.size();
        while (lo + 1 < hi) {
            const U64 mid = (lo + hi) / 2;
            if (visualRows[groupStartRows[mid]].top <= y) {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        return groupStartRows[lo];
    }

    U64 groupEndRow(U64 groupStartRow) const {
        const auto it = std::upper_bound(groupStartRows.begin(), groupStartRows.end(), groupStartRow);
        return it != groupStartRows.end() ? *it : visualRows.size();
    }

    U64 groupStartOf(U64 visualRow) const {
        const auto it = std::upper_bound(groupStartRows.begin(), groupStartRows.end(), visualRow);
        return it == groupStartRows.begin() ? 0 : *(it - 1);
    }

    U64 columnEndRow(U64 visualRow) const {
        U64 line = visualRows[visualRow].line + 1;
        while (line < visualRowStartIndex.size() && lineRowModeAt(line) == RowMode::Below) {
            ++line;
        }
        return line < visualRowStartIndex.size() ? visualRowStartIndex[line] : visualRows.size();
    }

    U64 columnBeginRow(U64 visualRow) const {
        U64 line = visualRows[visualRow].line;
        while (line > 0 && lineRowModeAt(line) == RowMode::Below) {
            --line;
        }
        return std::max(groupStartOf(visualRow), visualRowStartIndex[line]);
    }

    U64 rowInRangeAtY(U64 begin, U64 end, F32 y) const {
        U64 lo = begin;
        U64 hi = std::max(end, begin + 1);
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

    template<typename F>
    void forEachGroupColumn(U64 groupStartRow, F&& visit) const {
        const U64 groupEnd = groupEndRow(groupStartRow);
        U64 begin = groupStartRow;
        while (begin < groupEnd) {
            const U64 end = std::max(begin + 1, std::min(groupEnd, columnEndRow(begin)));
            visit(begin, end);
            begin = end;
        }
    }

    U64 visualRowNear(F32 contentY, F32 contentX) const {
        const U64 groupStart = visualRowAtContentY(contentY);
        U64 best = groupStart;
        F32 bestScore = std::numeric_limits<F32>::max();
        forEachGroupColumn(groupStart, [&](U64 begin, U64 end) {
            const U64 r = rowInRangeAtY(begin, end, contentY);
            const auto& candidate = visualRows[r];
            const F32 bottom = candidate.top + candidate.height;
            const bool containsY = contentY >= candidate.top && contentY < bottom;
            const F32 yDistance = contentY < candidate.top ? candidate.top - contentY
                                : contentY >= bottom ? contentY - bottom
                                                     : 0.0f;
            const F32 score = (containsY ? 0.0f : kHitContainsWeight) +
                              yDistance * kHitVerticalWeight +
                              contentDistanceX(candidate, contentX + lineScrollX(candidate.line));
            if (score < bestScore) {
                bestScore = score;
                best = r;
            }
        });
        return best;
    }

    U64 groupColumnNearX(U64 groupStartRow, F32 contentX, bool lastRow) const {
        U64 best = groupStartRow;
        F32 bestDistance = std::numeric_limits<F32>::max();
        forEachGroupColumn(groupStartRow, [&](U64 begin, U64 end) {
            const U64 r = lastRow ? end - 1 : begin;
            const F32 distance = contentDistanceX(visualRows[r], contentX);
            if (distance < bestDistance) {
                bestDistance = distance;
                best = r;
            }
        });
        return best;
    }

    F32 visibleTopContentY() const {
        const F32 clipTop = clip.has_value() ? std::max(rect.y, clip->y) : rect.y;
        return std::max(0.0f, contentYFromScreen(clipTop));
    }

    void collectVisibleRowRanges(F32 visibleTop, F32 visibleBottom) {
        visibleRowRanges.clear();
        if (visualRows.empty() || visibleBottom <= visibleTop) {
            return;
        }
        U64 groupStart = visualRowAtContentY(visibleTop);
        while (groupStart < visualRows.size() && visualRows[groupStart].top < visibleBottom) {
            forEachGroupColumn(groupStart, [&](U64 begin, U64 end) {
                U64 first = rowInRangeAtY(begin, end, visibleTop);
                while (first < end && visualRows[first].top + visualRows[first].height <= visibleTop) {
                    ++first;
                }
                U64 last = first;
                while (last < end && visualRows[last].top < visibleBottom) {
                    ++last;
                }
                if (last > first) {
                    visibleRowRanges.emplace_back(first, last);
                }
            });
            groupStart = groupEndRow(groupStart);
        }
    }

    F32 sourceLineTop(U64 line) const {
        ensureVisualRows();
        if (line < visualRowStartIndex.size()) {
            return rowTopPixels(visualRowStartIndex[line]);
        }
        return rowTopPixels(visualRows.size());
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
    F32 editorBottomPaddingPixels() const {
        if (!config.editable || !config.scrollPastEnd) {
            return 0.0f;
        }
        const auto padding = paddingPixels();
        return std::max(0.0f, rect.height - lineHeightPixels() - padding.top - padding.bottom - scrollbarBottomPixels());
    }
    F32 textContentHeightPixels() const {
        ensureVisualRows();
        return visualContentHeight;
    }
    F32 paddedContentHeightPixels() const {
        const auto padding = paddingPixels();
        return padding.top + textContentHeightPixels() + padding.bottom + scrollbarBottomPixels();
    }
    F32 contentHeightPixels() const {
        return paddedContentHeightPixels() + editorBottomPaddingPixels();
    }
    F32 boundedHeightPixels() const {
        if (config.minLines <= 0.0f && config.maxLines <= 0.0f) {
            return paddedContentHeightPixels();
        }
        const F32 lineHeight = lineHeightPixels();
        F32 lines = std::round(textContentHeightPixels() / lineHeight);
        if (config.minLines > 0.0f) lines = std::max(lines, config.minLines);
        if (config.maxLines > 0.0f) lines = std::min(lines, config.maxLines);
        return paddedContentHeightPixels() - textContentHeightPixels() + lines * lineHeight;
    }
    F32 maxScrollYPixels() const {
        const F32 range = contentHeightPixels() - rect.height;
        return range > kScrollRangeEpsilon ? range : 0.0f;
    }
    F32 maxLineAdvancePixels() const {
        F32 widest = contentMinWidthPixels();
        if (config.wrap != Wrap::None) {
            return widest;
        }
        for (U64 i = 0; i < lines.size(); ++i) {
            if (regionOf(i) == kNoRegion) {
                widest = std::max(widest, lineIndentAt(i) + lineWidthPixels(i));
            }
        }
        return widest;
    }
    F32 maxScrollXPixels() const {
        const F32 range = maxLineAdvancePixels() - textViewportWidthPixels();
        return range > kScrollRangeEpsilon ? range : 0.0f;
    }
    F32 contentWidthPixels() const {
        return (textLeftPixels() - rect.x) + maxLineAdvancePixels() + paddingPixels().right + scrollbarGutterPixels();
    }
    F32 measuredContentWidthPixels() const {
        ensureVisualRows();
        F32 widest = contentMinWidthPixels();
        for (const auto& row : visualRows) {
            const F32 width = rowContentWidth(row);
            const F32 trailing = width > 0.0f ? lineTrailingMarginPixels(row.line) : 0.0f;
            widest = std::max(widest, rowContentLeft(row) + width + lineRightInsetAt(row.line) + trailing);
        }
        return (textLeftPixels() - rect.x) + widest + paddingPixels().right + scrollbarGutterPixels();
    }
    Metrics computeMetrics() const {
        Metrics out;
        out.contentHeight = paddedContentHeightPixels();
        out.contentWidth = contentWidthPixels();
        out.scrollX = currentScrollX;
        out.scrollY = currentScrollY;
        out.overscroll = elastic;
        out.scrollbarGutter = scrollbarGutterPixels();
        out.padding = paddingPixels();
        out.sourceLines.resize(lines.size());
        for (U64 line = 0; line < lines.size(); ++line) {
            out.sourceLines[line] = {
                .top = sourceLineTop(line),
                .height = sourceLineHeight(line),
            };
        }
        out.regions.resize(regionScroll.size());
        for (U64 region = 0; region < regionScroll.size(); ++region) {
            out.regions[region] = {
                .scrollX = regionScroll[region],
                .maxScrollX = regionMaxScrollX(region),
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

    bool styleMetricsActive() const {
        return !config.monospace && static_cast<bool>(config.styler);
    }

    static bool IsIconCodepoint(U32 codepoint) {
        return codepoint >= 0xE000 && codepoint <= 0xF8FF;
    }

    bool iconsActive() const {
        return !config.iconFont.empty() && config.iconFont != config.fontName;
    }

    bool lineHasIcons(const std::string& line) const {
        if (!iconsActive() || Unicode::IsAscii(line)) {
            return false;
        }
        for (U64 i = 0; i < line.size();) {
            U32 codepoint = 0;
            i += Unicode::Decode(line, i, codepoint);
            if (IsIconCodepoint(codepoint)) {
                return true;
            }
        }
        return false;
    }

    std::string fontForRun(StyleId id, bool icon) const {
        return icon ? config.iconFont : fontForStyle(id);
    }

    U64 runEnd(const std::string& line, const std::vector<StyleId>& lineStyles,
               U64 start, U64 limit, bool& icon, bool fontBoundariesOnly = false) const {
        const auto styleAt = [&](U64 column) {
            return column < lineStyles.size() ? lineStyles[column] : 0;
        };
        const StyleId style = styleAt(start);
        U32 codepoint = 0;
        U64 pos = start + Unicode::Decode(line, start, codepoint);
        icon = iconsActive() && IsIconCodepoint(codepoint);
        const std::string font = fontForRun(style, icon);
        while (pos < limit) {
            const U64 length = Unicode::Decode(line, pos, codepoint);
            const bool iconAt = iconsActive() && IsIconCodepoint(codepoint);
            const bool boundary = fontBoundariesOnly
                ? fontForRun(styleAt(pos), iconAt) != font
                : styleAt(pos) != style || iconAt != icon;
            if (boundary) {
                break;
            }
            pos += length;
        }
        return std::min(pos, limit);
    }

    bool styleHasBackground(StyleId id) const {
        return id != 0 && id <= config.styleBackgroundColorKeys.size() &&
               !config.styleBackgroundColorKeys[id - 1].empty();
    }

    F32 stylePaddingPixels(StyleId id, F32 glyphSize) const {
        return styleMetricsActive() && styleHasBackground(id) ? glyphSize * Typography::StyleBackgroundPadRatio : 0.0f;
    }

    F32 scaleForStyle(StyleId id) const {
        if (!styleMetricsActive() || id == 0 || id > config.styleScales.size()) {
            return 1.0f;
        }
        const F32 scale = config.styleScales[id - 1];
        return scale > 0.0f ? scale : 1.0f;
    }

    ColorRGBA<F32> backgroundForStyle(StyleId id) const {
        if (id == 0 || id > theme.styleBackgrounds.size()) {
            return {0.0f, 0.0f, 0.0f, 0.0f};
        }
        return theme.styleBackgrounds[id - 1];
    }

    void ensureFontPools() {
        std::vector<std::string> wanted;
        auto fonts = config.styleFonts;
        if (iconsActive()) {
            fonts.push_back(config.iconFont);
        }
        for (const auto& font : fonts) {
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

    void ensureLanes() {
        static const std::vector<std::vector<StyleId>> kNoStyles;
        static const std::vector<StyleId> kNoLineStyles;
        const auto& regions = config.scrollRegions;
        const auto& styles = !regions.empty() && config.styler ? config.styler(lines, contentRevision) : kNoStyles;
        std::vector<std::pair<F32, F32>> groupKeys;
        std::vector<std::pair<U64, std::string>> wanted;
        regionGroup.assign(regions.size(), 0);
        groupRegion.clear();
        for (U64 region = 0; region < regions.size(); ++region) {
            const U64 first = regions[region].first;
            const std::pair<F32, F32> key = {lineIndentAt(first), lineRightInsetAt(first)};
            const auto found = std::find(groupKeys.begin(), groupKeys.end(), key);
            const U64 group = static_cast<U64>(found - groupKeys.begin());
            if (found == groupKeys.end()) {
                groupKeys.push_back(key);
                groupRegion.push_back(region);
            }
            regionGroup[region] = group;

            std::array<bool, 256> used = {};
            bool icons = false;
            for (U64 line = first; line <= regions[region].last && line < lines.size(); ++line) {
                const auto& lineStyles = line < styles.size() ? styles[line] : kNoLineStyles;
                used[0] = used[0] || lineStyles.size() < lines[line].size();
                for (const StyleId style : lineStyles) {
                    used[style] = true;
                }
                icons = icons || lineHasIcons(lines[line]);
            }
            const auto want = [&](std::string font) {
                std::pair<U64, std::string> lane = {group, std::move(font)};
                if (std::find(wanted.begin(), wanted.end(), lane) == wanted.end()) {
                    wanted.push_back(std::move(lane));
                }
            };
            for (U64 style = 0; style < used.size(); ++style) {
                if (used[style]) {
                    want(fontForStyle(static_cast<StyleId>(style)));
                }
            }
            if (icons) {
                want(config.iconFont);
            }
        }

        for (auto& lane : lanes) {
            lane.used = std::find(wanted.begin(), wanted.end(), std::pair{lane.group, lane.font}) != wanted.end();
        }
        for (const auto& [group, font] : wanted) {
            const auto covers = [&](const Lane& lane) {
                return lane.used && lane.group == group && lane.font == font;
            };
            if (std::any_of(lanes.begin(), lanes.end(), covers)) {
                continue;
            }
            auto free = std::find_if(lanes.begin(), lanes.end(), [](const Lane& lane) {
                return !lane.used;
            });
            if (free == lanes.end()) {
                lanes.push_back({.label = std::make_unique<Label>()});
                if (addChild) {
                    addChild(*lanes.back().label);
                }
                free = lanes.end() - 1;
            }
            free->group = group;
            free->font = font;
            free->used = true;
        }
    }

    std::vector<Label::Instance>& textPool(InstancePools& pools, const RowContext& ctx, const std::string& font) {
        if (ctx.clip) {
            for (U64 k = 0; k < lanes.size(); ++k) {
                if (lanes[k].used && lanes[k].group == ctx.group && lanes[k].font == font) {
                    return pools.laneText[k];
                }
            }
        }
        const U64 poolIndex = poolIndexForFont(font);
        return poolIndex == static_cast<U64>(-1) ? pools.text : pools.extraText[poolIndex];
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
        position.column = Unicode::Align(lines[position.line], position.column);
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

    SelectionState selectionState() const {
        SelectionState state;
        state.active = hasSelection();
        const auto range = selectionRange();
        state.first = range.first;
        state.second = range.second;
        state.matchText = singleLineSelectionText();
        return state;
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
            return {cursor.line, Unicode::PreviousCharacter(lines[cursor.line], cursor.column)};
        }
        if (cursor.line > 0) {
            return {cursor.line - 1, lines[cursor.line - 1].size()};
        }
        return cursor;
    }
    Position nextCharacterPosition() const {
        if (cursor.column < lines[cursor.line].size()) {
            return {cursor.line, Unicode::NextCharacter(lines[cursor.line], cursor.column)};
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

    void resetBlink() {
        blinkBase = now;
        if (layingOut) {
            return;
        }
        blinkPending = true;
        owner->invalidate(Dirty::Layout);
    }

    void replaceText() {
        lines = SplitLines(config.value);
        ++contentRevision;
        undoStack.clear();
        redoStack.clear();
        cursor = {0, 0};
        selectionAnchor = {0, 0};
        selectionActive = false;
        preferredContentX.reset();
        currentScrollX = 0.0f;
        std::fill(regionScroll.begin(), regionScroll.end(), 0.0f);
        currentScrollY = 0.0f;
        stickToBottomPending = false;
    }

    void refreshAfterCursorChange() {
        resetBlink();
        notifySelect();
        configureScroller();
        rebuildVisibleInstances();
    }

    void moveCursorTo(Position position, bool extendSelection, bool keepPreferredX = false) {
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
        if (!keepPreferredX) {
            preferredContentX.reset();
        }
        ensureCursorVisible();
        refreshAfterCursorChange();
    }

    U64 visualRowAfterVerticalMove(U64 fromRow, I64 delta, F32 contentX) const {
        U64 toRow = fromRow;
        for (I64 step = 0; step < std::abs(delta); ++step) {
            const U64 groupStart = groupStartOf(toRow);
            if (delta > 0) {
                const U64 columnEnd = std::min(groupEndRow(groupStart), columnEndRow(toRow));
                if (toRow + 1 < columnEnd) {
                    ++toRow;
                    continue;
                }
                const U64 nextGroup = groupEndRow(groupStart);
                if (nextGroup >= visualRows.size()) {
                    break;
                }
                toRow = groupColumnNearX(nextGroup, contentX, false);
            } else {
                if (toRow > columnBeginRow(toRow)) {
                    --toRow;
                    continue;
                }
                if (groupStart == 0) {
                    break;
                }
                toRow = groupColumnNearX(groupStartOf(groupStart - 1), contentX, true);
            }
        }
        return toRow;
    }

    void moveCursorVertically(I64 delta, bool extendSelection) {
        ensureVisualRows();
        const U64 fromRow = visualRowForPosition(cursor);
        if (!preferredContentX.has_value()) {
            preferredContentX = contentXAtColumn(visualRows[fromRow], cursor.column);
        }
        const U64 toRow = visualRowAfterVerticalMove(fromRow, delta, *preferredContentX);
        const auto& row = visualRows[toRow];
        const U64 column = std::min(columnAtContentX(row, *preferredContentX), rowLastColumn(row));
        moveCursorTo({row.line, column}, extendSelection, true);
    }

    void ensureCursorVisible() {
        cursorFollowPending = true;
        if (liveLayout) {
            resolveViewport(true);
        }
        const auto padding = paddingPixels();
        const U64 cursorRow = visualRowForPosition(cursor);
        const F32 rowTop = rowTopContent(cursorRow);
        const F32 rowBottom = rowTop + rowHeightPixels(cursorRow);
        if (rowTop < currentScrollY) {
            currentScrollY = rowTop;
        } else if (padding.top + rowBottom + padding.bottom > currentScrollY + rect.height) {
            currentScrollY = padding.top + rowBottom + padding.bottom - rect.height;
        }
        currentScrollY = std::clamp(currentScrollY, 0.0f, maxScrollYPixels());

        F32 cursorX = contentXAtColumn(visualRows[cursorRow], cursor.column);
        const U64 region = regionOf(cursor.line);
        if (region < regionScroll.size()) {
            const F32 left = regionLeft(region);
            const F32 width = regionRight(region) - left;
            const F32 margin = std::min(fontSizePixels * kCursorScrollMarginFontRatio, width * 0.5f);
            F32& offset = regionScroll[region];
            if (cursorX - left < offset + margin) {
                offset = std::max(0.0f, cursorX - left - margin);
            } else if (cursorX - left > offset + width - margin) {
                offset = std::max(0.0f, cursorX - left - width + margin);
            }
            offset = std::clamp(offset, 0.0f, regionMaxScrollX(region));
            cursorX -= offset;
        }
        const F32 viewportWidth = textViewportWidthPixels();
        const F32 margin = std::min(fontSizePixels * kCursorScrollMarginFontRatio, viewportWidth * 0.5f);
        if (cursorX < currentScrollX + margin) {
            currentScrollX = std::max(0.0f, cursorX - margin);
        } else if (cursorX > currentScrollX + viewportWidth - margin) {
            currentScrollX = std::max(0.0f, cursorX - viewportWidth + margin);
        }
        currentScrollX = std::clamp(currentScrollX, 0.0f, maxScrollXPixels());
    }

    void scrollToLine(U64 line) {
        if (liveLayout) {
            resolveViewport(true);
        }
        Position position = {line, 0};
        clampPosition(position);
        const F32 top = rowTopContent(visualRowForPosition(position)) - lineHeightPixels();
        currentScrollY = std::clamp(top, 0.0f, maxScrollYPixels());
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
    void contentChanged() {
        ++contentRevision;
        owner->invalidate(Dirty::Layout);
        ensureFontPools();
        ensureLanes();
        clampPosition(cursor);
        clampPosition(selectionAnchor);
        ensureCursorVisible();
        resetBlink();
        configureScroller();
        rebuildVisibleInstances();
        if (config.onChange) {
            config.onChange(textValue());
        }
    }
    void commitEdit() {
        preferredContentX.reset();
        contentChanged();
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
        if (config.masked) {
            insertText(std::string(1, character));
            return;
        }
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
            const bool pair = !config.masked && openingIndex != std::string_view::npos &&
                              cursor.column < line.size() && line[cursor.column] == kAutoPairClosings[openingIndex];
            U64 leadingWhitespace = LeadingWhitespaceColumn(line);
            if (!config.masked && !pair && leadingWhitespace == cursor.column) {
                const U64 previousIndent = ((cursor.column - 1) / kTabSize) * kTabSize;
                line.erase(previousIndent, cursor.column - previousIndent);
                cursor.column = previousIndent;
            } else {
                const U64 previous = Unicode::PreviousCharacter(line, cursor.column);
                line.erase(previous, cursor.column - previous + (pair ? 1 : 0));
                cursor.column = previous;
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
            line.erase(cursor.column, Unicode::NextCharacter(line, cursor.column) - cursor.column);
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

    void setSelection(Position anchor, Position end) {
        selectionAnchor = anchor;
        cursor = end;
        selectionActive = !(selectionAnchor == cursor);
        preferredContentX.reset();
        refreshAfterCursorChange();
    }

    void selectAll() {
        setSelection(documentStartPosition(), documentEndPosition());
    }

    void selectWordAt(Position position) {
        const auto range = wordRangeAt(position);
        if (!range.has_value()) {
            clampPosition(position);
            moveCursorTo(position, false);
            return;
        }
        setSelection(range->first, range->second);
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
        preferredContentX.reset();
        ensureCursorVisible();
        refreshAfterCursorChange();
    }

    void copySelectionOrLine() {
        if (config.masked) {
            return;
        }
        if (hasSelection()) {
            const auto [start, end] = selectionRange();
            SetClipboardText(textMetrics.window, textInRange(start, end));
        } else {
            SetClipboardText(textMetrics.window, lines[cursor.line] + "\n");
        }
    }

    void cutSelectionOrLine() {
        if (config.masked) {
            return;
        }
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
        contentChanged();
    }

    void redo() {
        if (redoStack.empty()) {
            return;
        }
        undoStack.push_back(snapshot());
        restoreSnapshot(redoStack.back());
        redoStack.pop_back();
        contentChanged();
    }

    bool hasFocus() const { return focused; }

    bool focus() {
        return focused || owner->requestFocus();
    }
    void gainFocusState() {
        focused = true;
        resetBlink();
        configureScroller();
        rebuildVisibleInstances();
    }
    void dropFocusState() {
        focused = false;
        mouseSelecting = false;
        clearSelection();
        configureScroller();
        rebuildVisibleInstances();
    }
    void clearFocus() {
        if (focused) {
            owner->releaseFocus();
        }
    }

    bool autoscrollSelection(const Extent2D<F32>& pixel) {
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
        const F32 previous = currentScrollY;
        if (std::abs(delta) > 1e-3f) {
            currentScrollY = std::clamp(currentScrollY + delta, 0.0f, maxScrollYPixels());
        }
        return currentScrollY != previous;
    }

    Position positionFromMouse(const Extent2D<F32>& pixel) const {
        ensureVisualRows();
        const F32 contentY = contentYFromScreen(pixel.y);
        const F32 contentX = contentXFromScreen(pixel.x);
        const U64 visualRowIndex = std::min<U64>(visualRowNear(contentY, contentX), visualRows.size() - 1);
        const auto& row = visualRows[visualRowIndex];
        return {row.line, columnAtContentX(row, contentX + lineScrollX(row.line))};
    }

    bool scrollRegion(MouseEvent& event) {
        const Extent2D<F32> position{event.position.x, event.position.y};
        if (regionScroll.empty() || !pointerInside(position)) {
            return false;
        }
        const bool swapped = event.modifiers.shift && event.scroll.x == 0.0f;
        const F32 dx = swapped ? event.scroll.y : event.scroll.x;
        const F32 dy = swapped ? 0.0f : event.scroll.y;
        if (dx == 0.0f || std::abs(dx) < std::abs(dy)) {
            return false;
        }
        ensureVisualRows();
        const F32 contentY = contentYFromScreen(position.y);
        const auto& row = visualRows[std::min<U64>(visualRowAtContentY(contentY), visualRows.size() - 1)];
        const U64 region = regionOf(row.line);
        if (region >= regionScroll.size() || contentY < row.top || contentY >= row.top + row.height) {
            return false;
        }
        const F32 step = WheelStep(event, lineHeightPixels() * kWheelScrollLines, pixelRatio);
        const F32 previous = regionScroll[region];
        const F32 target = previous - dx * step;
        const F32 next = std::clamp(target, 0.0f, regionMaxScrollX(region));
        if (next == previous) {
            return false;
        }
        event.scroll = {target == next ? 0.0f : (next - target) / step, 0.0f};
        event.modifiers.shift = false;
        regionScroll[region] = next;
        owner->invalidate(Dirty::Layout);
        return true;
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
                if (config.selectAllOnFocus && !hasFocus()) {
                    if (focus()) {
                        selectAll();
                    }
                    return true;
                }
                if (!focus()) {
                    return true;
                }
                const bool extend = event.modifiers.shift;
                if (event.clicks >= 2) {
                    selectWordAt(positionFromMouse(position));
                } else {
                    moveCursorTo(positionFromMouse(position), extend);
                    mouseSelecting = true;
                }
                return true;
            }
            case MouseEventType::Move: {
                if (!mouseSelecting) {
                    return false;
                }
                const bool scrolled = autoscrollSelection(position);
                const Position target = positionFromMouse(position);
                if (!scrolled && selectionActive && target == cursor) {
                    return true;
                }
                moveCursorTo(target, true);
                return true;
            }
            case MouseEventType::Cancel: {
                mouseSelecting = false;
                return false;
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

    struct Modifiers {
        bool command = false;
        bool control = false;
        bool alt = false;
        bool shift = false;
        bool shortcut() const { return command || control; }
        bool word() const { return alt || (control && !command); }
        bool none() const { return !command && !control && !alt; }
    };

    static Modifiers ModifiersOf(const KeyModifiers& modifiers) {
        return {
            .command = modifiers.super,
            .control = modifiers.control,
            .alt = modifiers.alt,
            .shift = modifiers.shift,
        };
    }

    std::string acceptedText(const std::string& text) const {
        std::string accepted;
        accepted.reserve(text.size());
        for (U64 pos = 0; pos < text.size();) {
            U32 codepoint = 0;
            const U64 length = std::max<U64>(1, Unicode::Decode(text, pos, codepoint));
            pos += length;
            if (config.singleLine && codepoint == '\r') {
                continue;
            }
            if (config.singleLine && codepoint == '\n') {
                codepoint = ' ';
            }
            if (config.characterFilter && !config.characterFilter(codepoint)) {
                continue;
            }
            accepted += Unicode::Encode(codepoint);
        }
        return accepted;
    }

    void submit() {
        if (config.onSubmit) {
            config.onSubmit(textValue());
        }
    }

    bool handleEscape() {
        if (config.onCancel) {
            config.onCancel();
            clearFocus();
            return true;
        }
        if (hasSelection()) {
            clearSelection();
            configureScroller();
            rebuildVisibleInstances();
            return true;
        }
        clearFocus();
        return false;
    }

    bool handleShortcut(KeyCode key, bool repeat, const Modifiers& mods) {
        if (key == KeyCode::A && !repeat) {
            selectAll();
            return true;
        }
        if (key == KeyCode::C && !repeat) {
            copySelectionOrLine();
            return true;
        }
        if (!config.editable) {
            return false;
        }
        if ((key == KeyCode::Enter || key == KeyCode::KeypadEnter) && !repeat && !mods.alt) {
            submit();
            return true;
        }
        if (key == KeyCode::X && !repeat) {
            cutSelectionOrLine();
            return true;
        }
        if (key == KeyCode::V && !repeat) {
            const std::string clipboardText = acceptedText(GetClipboardText(textMetrics.window));
            if (!clipboardText.empty()) {
                insertText(clipboardText);
            }
            return true;
        }
        if (key == KeyCode::Z) {
            if (mods.shift) {
                redo();
            } else {
                undo();
            }
            return true;
        }
        if (key == KeyCode::Y) {
            redo();
            return true;
        }
        return false;
    }

    bool handleEditing(KeyCode key, bool repeat, const Modifiers& mods) {
        if (key == KeyCode::Backspace) {
            if (mods.command) {
                deleteToPosition(lineStartPosition());
            } else if (mods.word()) {
                deleteToPosition(previousWordPosition());
            } else {
                backspace();
            }
            return true;
        }
        if (key == KeyCode::Delete) {
            if (mods.command) {
                deleteToPosition(lineEndPosition());
            } else if (mods.word()) {
                deleteToPosition(nextWordPosition());
            } else {
                deleteForward();
            }
            return true;
        }
        if (!mods.none()) {
            return false;
        }
        const bool enter = key == KeyCode::Enter || key == KeyCode::KeypadEnter;
        if (config.singleLine) {
            if (enter && !repeat && config.submitOnEnter && config.onSubmit) {
                submit();
                return true;
            }
            return false;
        }
        if (enter && config.submitOnEnter && !mods.shift) {
            if (repeat || !config.onSubmit) {
                return false;
            }
            submit();
            return true;
        }
        if (enter) {
            insertNewLine();
            return true;
        }
        if (key == KeyCode::Tab) {
            indentSelection(mods.shift);
            return true;
        }
        return false;
    }

    bool handleNavigation(KeyCode key, const Modifiers& mods) {
        switch (key) {
            case KeyCode::LeftArrow:
                if (hasSelection() && !mods.shift) {
                    moveCursorTo(selectionRange().first, false);
                } else {
                    moveCursorTo(mods.command ? lineStartPosition()
                                              : (mods.word() ? previousWordPosition() : previousCharacterPosition()),
                                 mods.shift);
                }
                return true;
            case KeyCode::RightArrow:
                if (mods.alt && mods.shift && !mods.command && !mods.control) {
                    expandSyntaxSelection();
                } else if (hasSelection() && !mods.shift) {
                    moveCursorTo(selectionRange().second, false);
                } else {
                    moveCursorTo(mods.command ? lineEndPosition()
                                              : (mods.word() ? nextWordPosition() : nextCharacterPosition()),
                                 mods.shift);
                }
                return true;
            case KeyCode::Home:
                moveCursorTo(smartLineStartPosition(), mods.shift);
                return true;
            case KeyCode::End:
                moveCursorTo(lineEndPosition(), mods.shift);
                return true;
            default:
                break;
        }
        if (config.singleLine) {
            return false;
        }
        switch (key) {
            case KeyCode::UpArrow:
                if (mods.command) {
                    moveCursorTo(documentStartPosition(), mods.shift);
                } else {
                    moveCursorVertically(-1, mods.shift);
                }
                return true;
            case KeyCode::DownArrow:
                if (mods.command) {
                    moveCursorTo(documentEndPosition(), mods.shift);
                } else {
                    moveCursorVertically(1, mods.shift);
                }
                return true;
            case KeyCode::PageUp:
                moveCursorVertically(-static_cast<I64>(pageStepLines()), mods.shift);
                return true;
            case KeyCode::PageDown:
                moveCursorVertically(static_cast<I64>(pageStepLines()), mods.shift);
                return true;
            default:
                return false;
        }
    }

    bool handleKey(const KeyEvent& event) {
        if (event.type != KeyEventType::Press || !hasFocus()) {
            return false;
        }
        const Modifiers mods = ModifiersOf(event.modifiers);
        if (event.key == KeyCode::Escape) {
            return !event.repeat && handleEscape();
        }
        if (mods.shortcut() && handleShortcut(event.key, event.repeat, mods)) {
            return true;
        }
        if (!config.editable) {
            return false;
        }
        return handleEditing(event.key, event.repeat, mods) || handleNavigation(event.key, mods);
    }

    bool handleText(const TextEvent& event) {
        if (!hasFocus() || !config.editable || !ModifiersOf(event.modifiers).none()) {
            return false;
        }
        const char32_t character = event.codepoint;
        if (config.characterFilter && !config.characterFilter(static_cast<U32>(character))) {
            return true;
        }
        if (character >= 32 && character < 127) {
            insertTypedCharacter(static_cast<char>(character));
        } else if (character >= 0xA0) {
            insertText(Unicode::Encode(static_cast<U32>(character)));
        }
        return true;
    }

    F64 nextBlink() const {
        const F64 phase = std::fmod(now - blinkBase, kCursorBlinkPeriod);
        return now + (phase < kCursorBlinkOnDuration ? kCursorBlinkOnDuration - phase : kCursorBlinkPeriod - phase);
    }

    bool caretActive() const {
        return config.editable && hasFocus() && windowFocused;
    }

    void reconcileCursorBlink() {
        if (!caretActive()) {
            return;
        }
        const bool blinkOn = std::fmod(now - blinkBase, kCursorBlinkPeriod) < kCursorBlinkOnDuration;
        if (blinkOn != lastBlinkOn) {
            rebuildVisibleInstances();
        }
    }

    void applyWidthLayout() {
        if (!config.widthLayout) {
            return;
        }
        WidthLayout next = config.widthLayout(textViewportWidthPixels());
        if (widthLayoutResolved && next == resolvedWidthLayout) {
            return;
        }
        resolvedWidthLayout = std::move(next);
        widthLayoutResolved = true;
        visualRowsValid = false;
        ensureLanes();
    }

    bool verticalScrollNeeded(bool layoutPass) const {
        if (!config.scrollbar) {
            return false;
        }
        if (layoutPass) {
            return maxScrollYPixels() > 0.0f;
        }
        if (config.maxLines > 0.0f) {
            return textContentHeightPixels() > config.maxLines * lineHeightPixels() + kScrollRangeEpsilon;
        }
        return config.editable && textContentHeightPixels() > lineHeightPixels() + kScrollRangeEpsilon;
    }

    void resolveViewport(bool layoutPass) {
        applyWidthLayout();
        const bool unchanged = viewportResolved && visualRowsValid &&
                               resolvedLayoutPass == layoutPass &&
                               resolvedRevision == contentRevision &&
                               resolvedSize.x == rect.width && resolvedSize.y == rect.height;
        if (unchanged) {
            return;
        }
        gutterReserved = false;
        bottomReserved = false;
        applyWidthLayout();
        bottomReserved = config.scrollbar && maxScrollXPixels() > 0.0f;
        if (verticalScrollNeeded(layoutPass)) {
            gutterReserved = true;
            applyWidthLayout();
            bottomReserved = config.scrollbar && maxScrollXPixels() > 0.0f;
        }
        ensureVisualRows();
        viewportResolved = true;
        resolvedLayoutPass = layoutPass;
        resolvedRevision = contentRevision;
        resolvedSize = {rect.width, rect.height};
    }

    void clampScroll() {
        currentScrollY = std::clamp(currentScrollY, 0.0f, maxScrollYPixels());
        currentScrollX = std::clamp(currentScrollX, 0.0f, maxScrollXPixels());
        for (U64 region = 0; region < regionScroll.size(); ++region) {
            regionScroll[region] = std::clamp(regionScroll[region], 0.0f, regionMaxScrollX(region));
        }
    }

    void configureScroller() {
        scroll.update({
            .id = config.id + ":scroll",
            .contentWidth = rect.width + maxScrollXPixels(),
            .contentHeight = rect.height + maxScrollYPixels(),
            .scrollX = currentScrollX,
            .scrollY = currentScrollY,
            .scrollbar = config.scrollbar,
            .wheelStep = lineHeightPixels() * kWheelScrollLines,
            .thickness = fontSizePixels * kScrollbarThicknessFontRatio,
            .margin = fontSizePixels * kScrollbarMarginFontRatio,
            .trackInset = config.scrollbarInset,
            .trackColorKey = config.scrollbarTrackColorKey,
            .thumbColorKey = config.scrollbarThumbColorKey,
            .onScrollY = [this](F32 scrollY) {
                currentScrollY = scrollY;
                owner->invalidate(Dirty::Layout);
            },
            .onScrollX = [this](F32 scrollX) {
                currentScrollX = scrollX;
                owner->invalidate(Dirty::Layout);
            },
            .onOverscroll = [this] {
                owner->requestFrame();
            },
        });
    }

    Rect clipped(const Rect& r) const {
        if (!clip.has_value()) {
            return r;
        }
        return Intersect(r, *clip);
    }

    FrameGeometry beginFrame() {
        ensureVisualRows();
        if (config.singleLine) {
            viewport.fixed(rect, clip.value_or(rect), 1);
        } else {
            viewport.update(rect, clip.value_or(rect), lineHeightPixels(), visibleLineCapacity(), surfaceHeight);
        }
        FrameGeometry frame;
        frame.rect = rect;
        frame.clip = viewport.bounds;
        frame.visible = !viewport.bounds.empty();
        frame.textClip = clipped(Rect{textClipLeftPixels(), rect.y,
                                      std::max(0.0f, textClipRightPixels() - textClipLeftPixels()), rect.height});
        frame.clipTop = viewport.bounds.y;
        frame.clipBottom = viewport.bounds.bottom();
        frame.clipLeft = textClipLeftPixels();
        frame.clipRight = textClipRightPixels();
        for (const U64 region : groupRegion) {
            const F32 left = screenXFromContent(regionLeft(region));
            const F32 right = screenXFromContent(regionRight(region));
            frame.groupClips.push_back(
                Intersect(*frame.textClip, {left, rect.y, std::max(0.0f, right - left), rect.height}));
        }
        frame.maxSegments = std::max<U64>(1, config.maxLineSegments);
        frame.lineCapacity = viewport.rowCapacity;
        frame.segmentCapacity = frame.lineCapacity * frame.maxSegments;
        if (frame.visible) {
            const F32 visibleTop = visibleTopContentY();
            collectVisibleRowRanges(visibleTop, visibleTop + (frame.clipBottom - frame.clipTop));
        } else {
            visibleRowRanges.clear();
        }
        return frame;
    }

    void rebuildVisibleInstances() {
        const FrameGeometry frame = beginFrame();
        InstancePools pools;
        pools.extraText.resize(extraFontNames.size());
        pools.laneText.resize(lanes.size());
        buildRows(frame, pools);
        uploadPools(frame, std::move(pools));
        updateChrome(frame);
        updateCaret(frame);
    }

    void buildRows(const FrameGeometry& frame, InstancePools& pools) {
        static const std::vector<std::vector<StyleId>> kNoStyles;
        static const std::vector<StyleId> kNoLineStyles;
        const auto& styles = config.styler ? config.styler(lines, contentRevision) : kNoStyles;
        const SelectionState selection = selectionState();

        for (const auto& [rangeBegin, rangeEnd] : visibleRowRanges) {
            for (U64 visualRow = rangeBegin; visualRow < rangeEnd; ++visualRow) {
                const VisualRow& row = visualRows[visualRow];
                const F32 rowTop = rowTopPixels(visualRow);
                if (rowTop >= frame.clipBottom || rowTop + row.height <= frame.clipTop) {
                    continue;
                }
                const U64 region = regionOf(row.line);
                const bool scrolled = region < regionScroll.size();
                const U64 group = scrolled ? regionGroup[region] : kNoRegion;
                const RowContext ctx{
                    .row = row,
                    .top = rowTop,
                    .height = row.height,
                    .line = lines[row.line],
                    .styles = row.line < styles.size() ? styles[row.line] : kNoLineStyles,
                    .glyphSize = lineGlyphSize(row.line),
                    .group = group,
                    .scrollX = scrolled ? regionScroll[region] : 0.0f,
                    .clip = group < frame.groupClips.size() ? &frame.groupClips[group] : nullptr,
                };
                buildTextSegments(frame, ctx, pools);
                buildLineNumber(frame, ctx, pools);
                buildSelection(frame, ctx, selection, pools);
                if (!config.masked) {
                    buildSelectionMatches(frame, ctx, selection, pools);
                }
            }
        }
    }

    U64 segmentEnd(const RowContext& ctx, U64 startColumn, bool lastSegment, bool& icon) const {
        const U64 scanLimit = std::min(ctx.row.end, startColumn + kTextSegmentCharacterCapacity);
        U64 endColumn = runEnd(ctx.line, ctx.styles, startColumn, scanLimit, icon, lastSegment);
        if (endColumn < ctx.row.end) {
            const U64 aligned = Unicode::Align(ctx.line, endColumn);
            if (aligned != endColumn) {
                const U64 previous = Unicode::PreviousCharacter(ctx.line, aligned);
                endColumn = previous > startColumn ? previous : aligned;
            }
        }
        return endColumn;
    }

    void buildTextSegments(const FrameGeometry& frame, const RowContext& ctx, InstancePools& pools) {
        const bool hasStyleBackgrounds = !theme.styleBackgrounds.empty();
        const U64 lineLen = ctx.line.size();
        const U64 runCapacity = frame.maxSegments;
        const U64 slotCapacity = frame.maxSegments * kSegmentLimitScale;
        U64 startColumn = ctx.row.start;
        U64 slots = 0;
        U64 runs = 0;
        bool merging = false;
        bool emitted = false;
        StyleId previousStyle = 0;
        bool previousIcon = false;
        std::string previousFont;
        while (startColumn < ctx.row.end) {
            const StyleId style = startColumn < ctx.styles.size() ? ctx.styles[startColumn] : 0;
            U32 codepoint = 0;
            Unicode::Decode(ctx.line, startColumn, codepoint);
            const bool startsWithIcon = iconsActive() && IsIconCodepoint(codepoint);
            const std::string font = fontForRun(style, startsWithIcon);
            const bool continues = emitted && font == previousFont &&
                                   (merging || (style == previousStyle && startsWithIcon == previousIcon));
            if ((!continues && runs == runCapacity) || slots == slotCapacity) {
                break;
            }
            const bool mergeRun = continues ? merging : runs + 1 == runCapacity;
            bool icon = false;
            const U64 endColumn = segmentEnd(ctx, startColumn, mergeRun, icon);
            const F32 segX = screenXFromContent(contentXAtColumn(ctx.row, startColumn)) - ctx.scrollX;
            const F32 segW = columnXInRow(ctx.row.line, startColumn, endColumn);
            if (segX > (ctx.clip ? ctx.clip->right() : frame.rect.right())) {
                break;
            }
            if (segX + segW < (ctx.clip ? ctx.clip->x : frame.rect.x)) {
                startColumn = endColumn;
                continue;
            }
            if (!continues) {
                ++runs;
                merging = mergeRun;
            }
            ++slots;
            emitted = true;
            previousStyle = style;
            previousIcon = icon;
            previousFont = font;
            const F32 glyphSize = ctx.glyphSize * scaleForStyle(style);
            const F32 pad = stylePaddingPixels(style, glyphSize);
            const bool spanStart = startColumn == 0 ||
                                   (startColumn - 1 < ctx.styles.size() ? ctx.styles[startColumn - 1] : 0) != style;
            const bool spanEnd = endColumn >= lineLen ||
                                 (endColumn < ctx.styles.size() ? ctx.styles[endColumn] : 0) != style;
            const F32 leftPad = spanStart ? pad : 0.0f;
            const F32 rightPad = spanEnd ? pad : 0.0f;
            auto& target = textPool(pools, ctx, fontForRun(style, icon));
            if (target.size() < frame.segmentCapacity) {
                target.push_back({
                    .rect = {segX + leftPad, ctx.top, std::max(0.0f, segW - leftPad - rightPad), ctx.height},
                    .str = config.masked ? MaskedRange(ctx.line, startColumn, endColumn)
                                         : ctx.line.substr(startColumn, endColumn - startColumn),
                    .visible = true,
                    .color = colorForStyle(style),
                    .fontSize = glyphSize,
                    .alignment = {0, 1},
                });
            }
            if (hasStyleBackgrounds) {
                const auto bg = backgroundForStyle(style);
                const F32 bgHeight = glyphSize * kStyleBackgroundHeightRatio;
                Rect bgRect = {segX, ctx.top + (ctx.height - bgHeight) * 0.5f, segW, bgHeight};
                if (ctx.clip) {
                    bgRect = Intersect(bgRect, {ctx.clip->x, bgRect.y, ctx.clip->width, bgRect.height});
                }
                if (bg.a > 0.0f && bgRect.width > 0.0f && pools.styleBackgrounds.size() < frame.segmentCapacity) {
                    pools.styleBackgrounds.push_back({
                        .rect = bgRect,
                        .visible = true,
                        .backgroundColor = bg,
                    });
                }
            }
            startColumn = endColumn;
        }
    }

    void buildLineNumber(const FrameGeometry& frame, const RowContext& ctx, InstancePools& pools) {
        if (!config.lineNumbers || ctx.row.start != 0 || lineContinuesRowAt(ctx.row.line) ||
            pools.lineNumbers.size() >= frame.lineCapacity) {
            return;
        }
        pools.lineNumbers.push_back({
            .rect = {frame.rect.x, ctx.top, lineNumberRightPixels(), ctx.height},
            .str = jst::fmt::format("{}", ctx.row.line + 1),
            .visible = true,
            .color = theme.lineNumber,
            .fontSize = contentFontSize(),
            .alignment = {2, 1},
        });
    }

    std::optional<Rect> rowRangeScreenRect(const FrameGeometry& frame, const RowContext& ctx,
                                           U64 a, U64 b, bool breakAtEnd) const {
        const auto& row = ctx.row;
        const U64 lineLen = ctx.line.size();
        const U64 s = std::max(a, row.start);
        const U64 e = std::min(b, row.end);
        const bool extend = breakAtEnd && row.end >= lineLen;
        if (e <= s && !extend) {
            return std::nullopt;
        }
        const F32 startX = screenXFromContent(contentXAtColumn(row, s)) - ctx.scrollX;
        F32 endX = screenXFromContent(contentXAtColumn(row, e)) - ctx.scrollX;
        if (extend) {
            endX = std::max(endX, screenXFromContent(contentXAtColumn(row, lineLen)) - ctx.scrollX +
                                      lineAdvancePixels(row.line));
        }
        const F32 clippedStart = std::max(startX, ctx.clip ? ctx.clip->x : frame.clipLeft);
        const F32 clippedEnd = std::min(endX, ctx.clip ? ctx.clip->right() : frame.clipRight);
        if (clippedEnd - clippedStart <= 0.0f) {
            return std::nullopt;
        }
        return Rect{clippedStart, ctx.top, clippedEnd - clippedStart, ctx.height};
    }

    void buildSelection(const FrameGeometry& frame, const RowContext& ctx,
                        const SelectionState& selection, InstancePools& pools) {
        const U64 line = ctx.row.line;
        if (!selection.active || line < selection.first.line || line > selection.second.line) {
            return;
        }
        const U64 lo = line == selection.first.line ? selection.first.column : 0;
        const U64 hi = line == selection.second.line ? selection.second.column : ctx.line.size();
        const bool breakAtEnd = line < selection.second.line;
        const auto r = rowRangeScreenRect(frame, ctx, lo, hi, breakAtEnd);
        if (r.has_value() && pools.selection.size() < frame.segmentCapacity) {
            pools.selection.push_back({.rect = *r, .backgroundColor = theme.selection});
        }
    }

    void buildSelectionMatches(const FrameGeometry& frame, const RowContext& ctx,
                               const SelectionState& selection, InstancePools& pools) {
        if (!selection.matchText.has_value()) {
            return;
        }
        const auto& needle = *selection.matchText;
        std::size_t found = ctx.line.find(needle);
        while (found != std::string::npos && pools.matches.size() < kSelectionMatchCapacity) {
            const U64 startCol = static_cast<U64>(found);
            const U64 endCol = startCol + needle.size();
            const bool isPrimary = selection.active && ctx.row.line == selection.first.line &&
                                   startCol == selection.first.column && endCol == selection.second.column;
            if (!isPrimary) {
                if (const auto r = rowRangeScreenRect(frame, ctx, startCol, endCol, false)) {
                    pools.matches.push_back({.rect = *r, .backgroundColor = theme.selectionMatch});
                }
            }
            found = ctx.line.find(needle, found + needle.size());
        }
    }

    void uploadPools(const FrameGeometry& frame, InstancePools&& pools) {
        const bool hasStyleBackgrounds = !theme.styleBackgrounds.empty();
        const U64 segmentCapacity = frame.segmentCapacity;
        selectionMatchBox.update({
            .instances = std::move(pools.matches),
            .clip = frame.textClip,
            .capacity = kSelectionMatchCapacity,
        });
        selectionBox.update({
            .instances = std::move(pools.selection),
            .clip = frame.textClip,
            .capacity = segmentCapacity,
        });
        styleBackgroundBox.update({
            .instances = std::move(pools.styleBackgrounds),
            .clip = frame.textClip,
            .cornerRadius = contentFontSize() * kStyleBackgroundCornerRatio,
            .capacity = hasStyleBackgrounds ? segmentCapacity : 0,
        });
        codeLabels.update({
            .instances = std::move(pools.text),
            .clip = frame.textClip,
            .fontName = config.fontName,
            .maxCharacters = kTextSegmentCharacterCapacity,
            .capacity = segmentCapacity,
        });
        for (U64 k = 0; k < extraFontLabels.size(); ++k) {
            const bool live = k < extraFontNames.size();
            extraFontLabels[k]->update({
                .instances = live ? std::move(pools.extraText[k]) : std::vector<Label::Instance>{},
                .clip = frame.textClip,
                .fontName = live ? extraFontNames[k] : config.fontName,
                .maxCharacters = kTextSegmentCharacterCapacity,
                .capacity = segmentCapacity,
            });
        }
        for (U64 k = 0; k < lanes.size(); ++k) {
            const auto& lane = lanes[k];
            lanes[k].label->update({
                .instances = std::move(pools.laneText[k]),
                .clip = lane.used && lane.group < frame.groupClips.size() ? frame.groupClips[lane.group]
                                                                          : frame.textClip,
                .fontName = lane.font,
                .maxCharacters = kTextSegmentCharacterCapacity,
                .capacity = segmentCapacity,
            });
        }
        numberLabels.update({
            .instances = std::move(pools.lineNumbers),
            .clip = frame.clip,
            .maxCharacters = kMaxLineNumberCharacters,
            .capacity = frame.lineCapacity,
        });
    }

    void updateChrome(const FrameGeometry& frame) {
        backgroundBox.update({
            .instances = {{.rect = frame.rect, .visible = frame.visible, .backgroundColor = theme.background}},
        });

        const U64 cursorRow = visualRowForPosition(cursor);
        const F32 activeTop = rowTopPixels(cursorRow);
        const F32 activeHeight = rowHeightPixels(cursorRow);
        activeLineBox.update({
            .instances = {{
                .rect = {frame.rect.x, activeTop, frame.rect.width, activeHeight},
                .visible = frame.visible && config.showActiveLine && hasFocus() && !hasSelection() &&
                            activeTop + activeHeight > frame.rect.y && activeTop < frame.rect.bottom(),
                .backgroundColor = theme.activeLine,
            }},
            .clip = frame.clip,
        });

        const F32 advance = cellAdvancePixels(contentFontSize());
        const F32 separatorWidth = std::max(1.0f, std::round(fontSizePixels * kSeparatorWidthFontRatio));
        const F32 separatorCenter = frame.rect.x + gutterWidthPixels() -
                                    kLineNumberRightPaddingCharacters * 0.5f * advance;
        gutterBox.update({
            .instances = {{
                .rect = {separatorCenter - separatorWidth * 0.5f, frame.rect.y, separatorWidth, frame.rect.height},
                .visible = frame.visible && config.lineNumbers,
                .backgroundColor = theme.gutterSeparator,
            }},
            .clip = frame.clip,
        });
    }

    void updateCaret(const FrameGeometry& frame) {
        const bool blinkOn = std::fmod(now - blinkBase, kCursorBlinkPeriod) < kCursorBlinkOnDuration;
        lastBlinkOn = blinkOn;
        const U64 cursorRow = visualRowForPosition(cursor);
        const F32 cursorWidth = std::max(1.0f, std::round(fontSizePixels * kCursorWidthFontRatio));
        const F32 cursorTop = rowTopPixels(cursorRow);
        const F32 cursorHeight = rowHeightPixels(cursorRow);
        const F32 cursorX = screenXFromContent(contentXAtColumn(visualRows[cursorRow], cursor.column)) -
                            lineScrollX(cursor.line);
        const U64 cursorRegion = regionOf(cursor.line);
        const U64 cursorGroup = cursorRegion < regionGroup.size() ? regionGroup[cursorRegion] : kNoRegion;
        cursorBox.update({
            .instances = {{
                .rect = {cursorX, cursorTop, cursorWidth, cursorHeight},
                .visible = frame.visible && caretActive() && blinkOn &&
                            cursorTop + cursorHeight > frame.rect.y && cursorTop < frame.rect.bottom(),
                .backgroundColor = theme.cursor,
            }},
            .clip = cursorGroup < frame.groupClips.size() ? std::optional<Rect>(frame.groupClips[cursorGroup])
                                                          : frame.textClip,
        });
    }
};

TextGrid::TextGrid() {
    this->impl = std::make_unique<Impl>();
    this->impl->owner = this;
    setClipsChildren(true);
    this->impl->addChild = [this](Component& child) {
        addChild(child);
    };
    addChild(this->impl->backgroundBox);
    addChild(this->impl->activeLineBox);
    addChild(this->impl->styleBackgroundBox);
    addChild(this->impl->selectionMatchBox);
    addChild(this->impl->selectionBox);
    addChild(this->impl->gutterBox);
    addChild(this->impl->numberLabels);
    addChild(this->impl->codeLabels);
    addChild(this->impl->cursorBox);
    addChild(this->impl->scroll);
    this->impl->ensureFontPools();
}

TextGrid::~TextGrid() = default;

void TextGrid::update(Config config) {
    if (config != impl->config) {
        invalidate(Dirty::Layout);
    }
    const bool valueChanged = impl->config.value != config.value;
    const bool regionsChanged = impl->config.scrollRegions != config.scrollRegions;
    const bool metricsChanged = impl->config.fontSize != config.fontSize ||
                                impl->config.fontScale != config.fontScale ||
                                impl->config.lineHeight != config.lineHeight ||
                                impl->config.styleScales != config.styleScales ||
                                impl->config.styleBackgroundColorKeys != config.styleBackgroundColorKeys ||
                                impl->config.styleRevision != config.styleRevision ||
                                impl->config.styleFonts != config.styleFonts ||
                                impl->config.fontName != config.fontName ||
                                impl->config.iconFont != config.iconFont ||
                                impl->config.monospace != config.monospace ||
                                impl->config.lineScale != config.lineScale ||
                                impl->config.lineTopGap != config.lineTopGap ||
                                impl->config.lineIndent != config.lineIndent ||
                                impl->config.lineRowMode != config.lineRowMode ||
                                impl->config.lineWrapWidth != config.lineWrapWidth ||
                                impl->config.lineRightInset != config.lineRightInset ||
                                impl->config.contentMinWidth != config.contentMinWidth ||
                                regionsChanged ||
                                static_cast<bool>(impl->config.widthLayout) != static_cast<bool>(config.widthLayout) ||
                                static_cast<bool>(impl->config.styler) != static_cast<bool>(config.styler);
    const bool viewportChanged = metricsChanged ||
                                 impl->config.wrap != config.wrap ||
                                 impl->config.padding != config.padding ||
                                 impl->config.scrollbar != config.scrollbar ||
                                 impl->config.scrollbarInset != config.scrollbarInset ||
                                 impl->config.scrollPastEnd != config.scrollPastEnd ||
                                 impl->config.minLines != config.minLines ||
                                 impl->config.maxLines != config.maxLines ||
                                 impl->config.editable != config.editable ||
                                 impl->config.lineNumbers != config.lineNumbers;
    const bool wasAtBottom = impl->rect.height <= 0.0f ||
                             impl->currentScrollY + 1.0f >= impl->maxScrollYPixels();
    impl->config = std::move(config);
    impl->fontSizePixels = impl->config.fontSize;

    if (viewportChanged) {
        impl->viewportResolved = false;
        ++impl->measureGeneration;
    }

    if (metricsChanged) {
        impl->visualRowsValid = false;
        impl->linePrefixesValid = false;
        impl->widthLayoutResolved = false;
        impl->viewportResolved = false;
        impl->cellAdvanceCache.clear();
        impl->preferredContentX.reset();
    }

    if ((valueChanged && impl->config.value != impl->textValue()) || impl->lines.empty()) {
        if (impl->config.editable) {
            impl->replaceText();
        } else {
            impl->lines = SplitLines(impl->config.value);
            ++impl->contentRevision;
            impl->clampPosition(impl->cursor);
            impl->clampPosition(impl->selectionAnchor);
            impl->stickToBottomPending = impl->config.stickToBottom && wasAtBottom;
        }
    }

    if (regionsChanged) {
        impl->regionScroll.assign(impl->config.scrollRegions.size(), 0.0f);
    }
    impl->ensureFontPools();
    if (valueChanged || metricsChanged) {
        impl->ensureLanes();
    }
}

TextGrid::Position TextGrid::cursor() const {
    return impl->cursor;
}

void TextGrid::setCursor(Position position) {
    impl->moveCursorTo(position, false);
    invalidate(Dirty::Layout);
}

bool TextGrid::focused() const {
    return impl->hasFocus();
}

void TextGrid::focus(bool selectAll) {
    if (!impl->focus()) {
        return;
    }
    if (selectAll) {
        impl->selectAll();
    }
    invalidate(Dirty::Layout);
}

void TextGrid::offerFocus(bool offered, bool selectAll) {
    impl->offered = offered;
    impl->selectAllOnOffer = offered && selectAll;
    Component::offerFocus(offered);
}

void TextGrid::blur() {
    impl->clearFocus();
    invalidate(Dirty::Layout);
}

void TextGrid::revert() {
    if (impl->textValue() == impl->config.value) {
        return;
    }
    impl->replaceText();
    impl->visualRowsValid = false;
    impl->linePrefixesValid = false;
    impl->widthLayoutResolved = false;
    impl->viewportResolved = false;
    impl->ensureFontPools();
    impl->ensureLanes();
    impl->rebuildVisibleInstances();
    invalidate(Dirty::Layout);
}

void TextGrid::moveCursorRows(I64 delta, bool extendSelection) {
    impl->moveCursorVertically(delta, extendSelection);
    invalidate(Dirty::Layout);
}

void TextGrid::scrollToLine(U64 line) {
    impl->scrollToLine(line);
    invalidate(Dirty::Layout);
}

const TextGrid::Metrics& TextGrid::metrics() const {
    return this->impl->storedMetrics;
}

Extent2D<F32> TextGrid::measure(const Context& ctx, Extent2D<F32> available) {
    impl->beginMetricsFrame(ctx.render);

    const auto& memo = impl->measureMemo;
    if (memo.valid && memo.availableWidth == available.x && memo.height == impl->rect.height &&
        memo.revision == impl->contentRevision && memo.generation == impl->measureGeneration &&
        memo.window == ctx.render) {
        impl->lastNaturalWidth = memo.naturalWidth;
        return FitExtent(memo.result, available);
    }

    const F32 maxWidth = available.x;
    const Rect savedRect = impl->rect;
    impl->rect = {0.0f, 0.0f, maxWidth, impl->rect.height};
    impl->ensureFontPools();
    impl->resolveViewport(false);
    const F32 height = impl->boundedHeightPixels();
    impl->lastNaturalWidth = impl->measuredContentWidthPixels();
    const F32 width = std::min(maxWidth, impl->lastNaturalWidth);
    impl->rect = savedRect;
    if (impl->liveLayout) {
        impl->resolveViewport(true);
    }

    impl->measureMemo = {
        .valid = impl->fontsReady(),
        .availableWidth = available.x,
        .height = savedRect.height,
        .revision = impl->contentRevision,
        .generation = impl->measureGeneration,
        .window = ctx.render,
        .result = {width, height},
        .naturalWidth = impl->lastNaturalWidth,
    };
    return FitExtent({width, height}, available);
}

F32 TextGrid::naturalWidth() const {
    return impl->lastNaturalWidth;
}

void TextGrid::layout(const Context& ctx) {
    impl->rect = frame();
    impl->clip = std::optional<Rect>(Intersect(frame(), clip()));
    impl->surfaceHeight = static_cast<F32>(ctx.framebufferSize.y);
    impl->pixelRatio = ctx.pixelRatio;
    impl->now = ctx.time;
    impl->layingOut = true;
    if (ctx.windowFocused && !impl->windowFocused) {
        impl->blinkPending = true;
    }
    impl->windowFocused = ctx.windowFocused;
    if (impl->blinkPending) {
        impl->blinkBase = impl->now;
        impl->blinkPending = false;
    }

    impl->beginMetricsFrame(ctx.render);
    impl->resolveTheme(ctx);
    impl->ensureFontPools();
    impl->resolveViewport(true);
    impl->liveLayout = true;
    if (impl->cursorFollowPending) {
        impl->cursorFollowPending = false;
        impl->ensureCursorVisible();
        impl->cursorFollowPending = false;
    }
    if (impl->stickToBottomPending) {
        impl->currentScrollY = impl->maxScrollYPixels();
        impl->stickToBottomPending = false;
    }
    impl->clampScroll();
    impl->configureScroller();
    layoutChild(ctx, impl->scroll, frame());
    impl->elastic = impl->scroll.overscroll(ctx);
    impl->rebuildVisibleInstances();

    impl->reconcileCursorBlink();
    impl->notifyLayout();
    if (impl->caretActive()) {
        requestFrame(impl->nextBlink());
    }
    impl->layingOut = false;
}

bool TextGrid::keyEvent(const KeyEvent& event) {
    return impl->handleKey(event);
}

bool TextGrid::textEvent(const TextEvent& event) {
    return impl->handleText(event);
}

void TextGrid::focusChanged(bool focused) {
    if (focused) {
        impl->gainFocusState();
        if (impl->selectAllOnOffer) {
            impl->selectAll();
        } else if (impl->offered) {
            impl->moveCursorTo(impl->documentEndPosition(), false);
        }
    } else {
        impl->dropFocusState();
    }
    impl->offered = false;
    impl->selectAllOnOffer = false;
    invalidate(Dirty::Layout);
    if (impl->config.onFocusChange) {
        impl->config.onFocusChange(focused);
    }
}

bool TextGrid::editsText() const {
    return impl->config.editable;
}

std::optional<SurfaceCursor> TextGrid::cursorAt(const Extent2D<F32>& point) const {
    if (!impl->pointerInside(point)) {
        return std::nullopt;
    }
    const bool link = impl->config.isPositionInteractive &&
                      impl->config.isPositionInteractive(impl->positionFromMouse(point));
    return link ? SurfaceCursor::Hand : SurfaceCursor::TextInput;
}

bool TextGrid::mouseEvent(const MouseEvent& event) {
    if (event.type == MouseEventType::Scroll) {
        MouseEvent rest = event;
        if (!impl->scrollRegion(rest)) {
            return mouseEventChildren(event);
        }
        if (rest.scroll.x != 0.0f) {
            mouseEventChildren(rest);
        }
        return true;
    }
    if (mouseEventChildren(event)) {
        return true;
    }
    return impl->handleMouse(event);
}

}  // namespace Jetstream::Sakura
