#include <jetstream/render/sakura/components/retained/text_markdown.hh>

#include <jetstream/platform.hh>
#include <jetstream/render/sakura/components/retained/box.hh>
#include <jetstream/render/sakura/components/retained/label.hh>
#include <jetstream/render/sakura/components/retained/text_grid.hh>

#include "../../context.hh"
#include "../../retained/helpers.hh"

#include <algorithm>
#include <string>
#include <utility>
#include <vector>

namespace Jetstream::Sakura::Retained {

namespace {

constexpr F32 kHeadingScale[3] = {1.5f, 1.3f, 1.15f};
constexpr F32 kLineHeightRatio = 1.15f;
constexpr F32 kPaddingRatio = 6.0f / 15.0f;
constexpr F32 kParagraphGapRatio = 0.6f;
constexpr F32 kHeadingGapRatio = 1.0f;
constexpr F32 kAfterHeadingGapRatio = 0.2f;
constexpr F32 kListGapRatio = 0.15f;
constexpr F32 kIndentEmRatio = 1.4f;
constexpr F32 kCodePadRatio = 0.4f;
constexpr F32 kRuleThicknessRatio = 0.12f;
constexpr F32 kQuoteBarRatio = 0.18f;
constexpr F32 kQuoteIndentEmRatio = 0.4f;
constexpr F32 kScrollbarGutterFontRatio = 14.0f / 15.0f;
constexpr const char* kBodyFont = "default_body";

using StyleId = TextGrid::StyleId;
constexpr StyleId kStyleBold = 1;
constexpr StyleId kStyleItalic = 2;
constexpr StyleId kStyleBoldItalic = 3;
constexpr StyleId kStyleCode = 4;
constexpr StyleId kStyleLink = 5;

constexpr U64 kMaxDecorations = 256;
constexpr const char* kCodeFont = "default_mono";

struct Block {
    std::string text;
    F32 fontSize = 15.0f;
    F32 topGap = 0.0f;
    StyleId baseStyle = 0;
    F32 indent = 0.0f;
    std::string marker;
    bool codeBlock = false;
    bool quote = false;
    bool rule = false;
    bool bullet = false;
    bool inlineParse = true;
};

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

U64 ListMarker(const std::string& line, std::string& marker) {
    marker.clear();
    U64 i = 0;
    while (i < line.size() && line[i] == ' ') {
        ++i;
    }
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

std::string TrimRight(std::string s) {
    while (!s.empty() && (s.back() == '\r' || s.back() == ' ' || s.back() == '\t')) {
        s.pop_back();
    }
    return s;
}

std::vector<std::string> SplitLines(const std::string& value) {
    std::vector<std::string> lines;
    std::string line;
    for (const char c : value) {
        if (c == '\n') {
            lines.push_back(std::move(line));
            line.clear();
        } else {
            line += c;
        }
    }
    lines.push_back(std::move(line));
    return lines;
}

struct LinkSpan {
    U64 start = 0;
    U64 end = 0;
    std::string url;
};

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

    U64 i = 0;
    while (i < n) {
        const char c = line[i];
        if (c == '`') {
            const U64 j = line.find('`', i + 1);
            if (j != std::string::npos) {
                emitRange(i + 1, j, kStyleCode);
                i = j + 1;
                continue;
            }
        }
        if (line.compare(i, 3, "***") == 0) {
            const U64 j = line.find("***", i + 3);
            if (j != std::string::npos) {
                emitRange(i + 3, j, kStyleBoldItalic);
                i = j + 3;
                continue;
            }
        }
        if (line.compare(i, 2, "**") == 0) {
            const U64 j = line.find("**", i + 2);
            if (j != std::string::npos) {
                emitRange(i + 2, j, kStyleBold);
                i = j + 2;
                continue;
            }
        }
        if (line.compare(i, 2, "__") == 0) {
            const U64 j = line.find("__", i + 2);
            if (j != std::string::npos) {
                emitRange(i + 2, j, kStyleBold);
                i = j + 2;
                continue;
            }
        }
        if (c == '*' || c == '_') {
            const U64 j = line.find(c, i + 1);
            if (j != std::string::npos && j > i + 1) {
                emitRange(i + 1, j, kStyleItalic);
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
                    emitRange(i + 1, close, kStyleLink);
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

}  // namespace

struct TextMarkdown::Impl {
    struct Deco {
        U64 first = 0;
        U64 last = 0;
        bool code = false;
        bool quote = false;
        bool rule = false;
        bool bullet = false;
        std::string marker;
        F32 indent = 0.0f;
    };

    Config config;
    TextGrid grid;
    Box decorations;
    Label markers;
    std::vector<std::vector<StyleId>> combinedStyles;
    std::vector<std::vector<LinkSpan>> combinedLinks;
    std::vector<Deco> decos;
    std::string plainValue;
    std::vector<F32> lineScale;
    std::vector<F32> lineTopGap;
    std::vector<F32> lineIndent;
    bool parsed = false;

    const std::string* linkAt(TextGrid::Position pos) const {
        if (pos.line >= combinedLinks.size()) {
            return nullptr;
        }
        for (const auto& span : combinedLinks[pos.line]) {
            if (pos.column >= span.start && pos.column < span.end) {
                return &span.url;
            }
        }
        return nullptr;
    }

    std::vector<Block> parse() const {
        const F32 body = config.fontSize;
        const F32 lineHeight = body * kLineHeightRatio;
        const F32 indentUnit = body * kIndentEmRatio;
        std::vector<Block> blocks;
        std::vector<std::string> paragraph, quoteLines, codeLines;
        bool prevHeading = false;
        bool prevList = false;
        bool inCode = false;

        const auto gapAbove = [&](F32 ratio) {
            return blocks.empty() ? 0.0f : lineHeight * (prevHeading ? kAfterHeadingGapRatio : ratio);
        };
        const auto join = [](const std::vector<std::string>& v) {
            std::string s;
            for (U64 i = 0; i < v.size(); ++i) {
                if (i) {
                    s += '\n';
                }
                s += v[i];
            }
            return s;
        };
        const auto flushParagraph = [&]() {
            if (paragraph.empty()) {
                return;
            }
            Block b;
            b.text = join(paragraph);
            b.fontSize = body;
            b.topGap = gapAbove(kParagraphGapRatio);
            blocks.push_back(std::move(b));
            prevHeading = prevList = false;
            paragraph.clear();
        };
        const auto flushQuote = [&]() {
            if (quoteLines.empty()) {
                return;
            }
            Block b;
            b.text = join(quoteLines);
            b.fontSize = body;
            b.topGap = gapAbove(kParagraphGapRatio);
            b.indent = body * kQuoteIndentEmRatio;
            b.quote = true;
            blocks.push_back(std::move(b));
            prevHeading = prevList = false;
            quoteLines.clear();
        };
        const auto flushCode = [&]() {
            Block b;
            b.text = join(codeLines);
            b.fontSize = body;
            b.topGap = gapAbove(kParagraphGapRatio);
            b.codeBlock = true;
            b.baseStyle = kStyleCode;
            b.inlineParse = false;
            blocks.push_back(std::move(b));
            prevHeading = prevList = false;
            codeLines.clear();
        };

        for (const auto& raw : SplitLines(config.value)) {
            if (inCode) {
                if (raw.rfind("```", 0) == 0) {
                    inCode = false;
                    flushCode();
                } else {
                    codeLines.push_back(raw);
                }
                continue;
            }
            const std::string line = TrimRight(raw);
            if (line.rfind("```", 0) == 0) {
                flushParagraph();
                flushQuote();
                inCode = true;
                continue;
            }
            if (IsRule(line)) {
                flushParagraph();
                flushQuote();
                Block b;
                b.fontSize = body;
                b.topGap = gapAbove(kParagraphGapRatio);
                b.rule = true;
                blocks.push_back(std::move(b));
                prevHeading = prevList = false;
                continue;
            }
            const U64 level = HeadingLevel(line);
            if (level > 0) {
                flushParagraph();
                flushQuote();
                const U64 scale = std::min<U64>(level, 3);
                const F32 gap = blocks.empty() ? 0.0f
                              : lineHeight * (prevHeading ? kAfterHeadingGapRatio : kHeadingGapRatio);
                Block b;
                b.text = line.substr(level + 1);
                b.fontSize = body * kHeadingScale[scale - 1];
                b.topGap = gap;
                b.baseStyle = kStyleBold;
                blocks.push_back(std::move(b));
                prevHeading = true;
                prevList = false;
                continue;
            }
            if (!line.empty() && line[0] == '>') {
                flushParagraph();
                U64 s = 1;
                while (s < line.size() && line[s] == ' ') {
                    ++s;
                }
                quoteLines.push_back(line.substr(s));
                continue;
            }
            flushQuote();
            std::string marker;
            U64 lead = 0;
            while (lead < line.size() && line[lead] == ' ') {
                ++lead;
            }
            const U64 contentStart = ListMarker(line, marker);
            if (contentStart > 0) {
                flushParagraph();
                Block b;
                b.text = line.substr(contentStart);
                b.fontSize = body;
                b.topGap = lineHeight * (prevList ? kListGapRatio
                                       : prevHeading ? kAfterHeadingGapRatio
                                                     : kParagraphGapRatio);
                b.indent = indentUnit * static_cast<F32>(lead / 2 + 1);
                b.marker = marker;
                b.bullet = marker.empty();
                blocks.push_back(std::move(b));
                prevHeading = false;
                prevList = true;
                continue;
            }
            if (line.empty()) {
                flushParagraph();
                continue;
            }
            paragraph.push_back(line);
        }
        flushParagraph();
        flushQuote();
        if (inCode) {
            flushCode();
        }
        return blocks;
    }

    void rebuild() {
        const std::vector<Block> blocks = parse();

        const F32 body = config.fontSize;
        const F32 codePad = body * kCodePadRatio;

        std::vector<std::string> lines;
        lineScale.clear();
        lineTopGap.clear();
        lineIndent.clear();
        combinedStyles.clear();
        combinedLinks.clear();
        decos.clear();

        const auto addLine = [&](std::string text, F32 scale, F32 gap, F32 indent,
                                 std::vector<StyleId> styles, std::vector<LinkSpan> links) {
            lines.push_back(std::move(text));
            lineScale.push_back(scale);
            lineTopGap.push_back(gap);
            lineIndent.push_back(indent);
            combinedStyles.push_back(std::move(styles));
            combinedLinks.push_back(std::move(links));
        };

        for (const auto& b : blocks) {
            const F32 scale = b.fontSize / body;
            const F32 indent = b.indent + (b.codeBlock ? codePad : 0.0f);
            const U64 firstLine = lines.size();

            if (b.rule) {
                addLine("", 1.0f, b.topGap, 0.0f, {}, {});
                decos.push_back({firstLine, firstLine, false, false, true, false, "", 0.0f});
                continue;
            }

            const auto sources = SplitLines(b.text);
            for (U64 li = 0; li < sources.size(); ++li) {
                const F32 gap = li == 0 ? b.topGap : 0.0f;
                std::vector<StyleId> styles;
                std::vector<LinkSpan> links;
                std::string display;
                if (b.inlineParse) {
                    ParseInline(sources[li], b.baseStyle, display, styles, links);
                } else {
                    display = sources[li];
                    styles.assign(display.size(), b.baseStyle);
                }
                addLine(std::move(display), scale, gap, indent, std::move(styles), std::move(links));
            }
            const U64 lastLine = lines.empty() ? firstLine : lines.size() - 1;

            if (b.codeBlock) {
                decos.push_back({firstLine, lastLine, true, false, false, false, "", indent});
            } else if (b.quote) {
                decos.push_back({firstLine, lastLine, false, true, false, false, "", indent});
            } else if (b.bullet || !b.marker.empty()) {
                decos.push_back({firstLine, firstLine, false, false, false, b.bullet, b.marker, indent});
            }
        }
        if (lines.empty()) {
            addLine("", 1.0f, 0.0f, 0.0f, {}, {});
        }

        plainValue.clear();
        for (U64 i = 0; i < lines.size(); ++i) {
            if (i) {
                plainValue += '\n';
            }
            plainValue += lines[i];
        }
    }

    TextGrid::Config buildGridConfig() {
        return {
            .id = config.id + ":grid",
            .value = plainValue,
            .editable = false,
            .fontSize = config.fontSize,
            .fontName = kBodyFont,
            .monospace = false,
            .showActiveLine = false,
            .scrollbar = config.scrollbar,
            .wrap = TextGrid::Wrap::Word,
            .lineScale = lineScale,
            .lineTopGap = lineTopGap,
            .lineIndent = lineIndent,
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
            .styleColorKeys = config.styleColorKeys,
            .styleFonts = config.styleFonts,
            .styleBackgroundColorKeys = config.styleBackgroundColorKeys,
            .maxLineSegments = 16,
            .styler = [this](const std::vector<std::string>&, U64)
                          -> const std::vector<std::vector<StyleId>>& {
                return combinedStyles;
            },
            .onPositionClick = [this](TextGrid::Position pos) -> bool {
                if (const std::string* url = linkAt(pos)) {
                    (void)Platform::OpenUrl(*url);
                    return true;
                }
                return false;
            },
            .isPositionInteractive = [this](TextGrid::Position pos) -> bool {
                return linkAt(pos) != nullptr;
            },
        };
    }
};

TextMarkdown::TextMarkdown() {
    this->impl = std::make_unique<Impl>();
    setClipsChildren(true);
    add(this->impl->decorations);
    add(this->impl->grid);
    add(this->impl->markers);
}

TextMarkdown::~TextMarkdown() = default;

bool TextMarkdown::update(Config config) {
    const bool changed = !impl->parsed ||
                         impl->config.value != config.value ||
                         impl->config.fontSize != config.fontSize;
    impl->config = std::move(config);
    if (changed) {
        impl->rebuild();
        impl->parsed = true;
        invalidate(Dirty::Paint);
    }
    return true;
}

Extent2D<F32> TextMarkdown::measure(const Context& ctx, Extent2D<F32> available) {
    impl->grid.update(impl->buildGridConfig());
    return measureChild(impl->grid, ctx, available);
}

void TextMarkdown::layout(const Context& ctx) {
    const Rect bounds = frame();
    const Rect rect = bounds;
    const Rect clipPixel = Intersect(bounds, clip());

    impl->grid.update(impl->buildGridConfig());
    layoutChild(ctx, impl->grid, bounds);

    const auto& metrics = impl->grid.metrics();
    const auto sourceLineTop = [&](U64 line) -> F32 {
        return line < metrics.sourceLines.size() ? metrics.sourceLines[line].top : rect.y;
    };
    const auto sourceLineHeight = [&](U64 line) -> F32 {
        return line < metrics.sourceLines.size() ? metrics.sourceLines[line].height : 0.0f;
    };

    const F32 body = impl->config.fontSize;
    const F32 codePad = body * kCodePadRatio;
    const F32 bodyPad = body * kPaddingRatio;
    const F32 lineHeight = body * kLineHeightRatio;
    const F32 barWidth = body * kQuoteBarRatio;
    const F32 ruleThick = std::max(1.0f, body * kRuleThicknessRatio);
    const F32 dotSize = std::max(2.0f, body * 0.22f);
    const bool scrollbarVisible = impl->config.scrollbar && metrics.contentHeight > rect.height;
    const F32 decorationWidth = std::max(
        0.0f, rect.width - (scrollbarVisible ? body * kScrollbarGutterFontRatio : 0.0f));
    const bool on = !rect.empty();
    const auto codeBg = ctx.color(impl->config.scrollbarTrackColorKey);
    const auto barColor = ctx.color(impl->config.lineNumberColorKey);
    const auto ruleColor = ctx.color(impl->config.gutterSeparatorColorKey);
    const auto markerColor = ctx.color(impl->config.lineNumberColorKey);

    std::vector<Box::Instance> decoInstances;
    std::vector<Label::Instance> markerInstances;
    for (const auto& d : impl->decos) {
        if (decoInstances.size() >= kMaxDecorations || markerInstances.size() >= kMaxDecorations) {
            break;
        }
        const F32 top = sourceLineTop(d.first);

        if (d.rule) {
            decoInstances.push_back({
                .rect = {rect.x, top + lineHeight * 0.5f - ruleThick * 0.5f, decorationWidth, ruleThick},
                .visible = on,
                .backgroundColor = ruleColor,
            });
            continue;
        }
        if (d.code) {
            const F32 bottom = sourceLineTop(d.last) + sourceLineHeight(d.last);
            decoInstances.push_back({
                .rect = {rect.x, top - codePad, decorationWidth, (bottom - top) + 2.0f * codePad},
                .visible = on,
                .backgroundColor = codeBg,
            });
            continue;
        }
        if (d.quote) {
            const F32 bottom = sourceLineTop(d.last) + sourceLineHeight(d.last);
            decoInstances.push_back({
                .rect = {rect.x, top, barWidth, bottom - top},
                .visible = on,
                .backgroundColor = barColor,
            });
            continue;
        }
        const F32 textStart = rect.x + bodyPad + d.indent;
        if (d.bullet) {
            decoInstances.push_back({
                .rect = {textStart - body * 0.75f - dotSize, top + lineHeight * 0.5f - dotSize * 0.5f,
                         dotSize, dotSize},
                .visible = on,
                .backgroundColor = markerColor,
            });
        }
        if (!d.marker.empty()) {
            const F32 markerRight = textStart - body * 0.45f;
            const F32 markerWidth = body * 2.0f;
            markerInstances.push_back({
                .rect = {markerRight - markerWidth, top, markerWidth, lineHeight},
                .str = d.marker,
                .visible = on,
                .color = markerColor,
                .fontSize = body,
                .alignment = {2, 1},
            });
        }
    }

    impl->decorations.update({
        .id = impl->config.id + ":deco",
        .instances = std::move(decoInstances),
        .clip = clipPixel,
        .cornerRadius = std::max(1.0f, body * 0.4f),
        .capacity = kMaxDecorations,
    });
    impl->markers.update({
        .id = impl->config.id + ":markers",
        .instances = std::move(markerInstances),
        .clip = clipPixel,
        .fontName = kCodeFont,
        .maxCharacters = 8,
        .capacity = kMaxDecorations,
    });

    layoutChild(ctx, impl->decorations, bounds);
    layoutChild(ctx, impl->markers, bounds);

}

}  // namespace Jetstream::Sakura::Retained
