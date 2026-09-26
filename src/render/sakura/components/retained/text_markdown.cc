#include <jetstream/render/sakura/components/retained/text_markdown.hh>

#include <jetstream/platform.hh>
#include <jetstream/render/sakura/components/retained/box.hh>
#include <jetstream/render/sakura/components/retained/label.hh>
#include <jetstream/render/sakura/components/retained/text_grid.hh>

#include "../../context.hh"
#include "../../retained/helpers.hh"
#include "../../retained/text_lines.hh"

#include <algorithm>
#include <array>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace Jetstream::Sakura::Retained {

namespace {

constexpr F32 kLineHeightRatio = Typography::BodyLineHeight;
constexpr F32 kParagraphGapRatio = 0.6f;
constexpr F32 kHeadingGapRatio = 1.0f;
constexpr F32 kAfterHeadingGapRatio = 0.2f;
constexpr F32 kListGapRatio = 0.15f;
constexpr F32 kIndentEmRatio = 1.4f;
constexpr F32 kCodePadRatio = 0.4f;
constexpr F32 kRuleThicknessRatio = 0.12f;
constexpr F32 kRuleRowScale = 0.5f;
constexpr F32 kQuoteBarRatio = 0.18f;
constexpr F32 kQuoteIndentEmRatio = 0.4f;
constexpr F32 kScrollbarGutterFontRatio = 14.0f / 15.0f;
constexpr F32 kBulletDotRatio = 0.22f;
constexpr F32 kBulletGapEmRatio = 0.75f;
constexpr F32 kMarkerGapEmRatio = 0.45f;
constexpr F32 kMarkerWidthEmRatio = 2.0f;
constexpr F32 kDecorationCornerRatio = 0.4f;
constexpr U64 kMaxDecorations = 256;
constexpr U64 kMaxMarkerCharacters = 8;
constexpr U64 kGridLineSegments = 32;
constexpr const char* kBodyFont = TextMarkdown::BodyFont;
constexpr const char* kCodeFont = Typography::MonoFont;

using StyleId = TextGrid::StyleId;
using Style = TextMarkdown::Style;

constexpr std::array<std::pair<std::string_view, StyleId>, 4> kInlineDelimiters = {{
    {"`", Style::Code},
    {"***", Style::BoldItalic},
    {"**", Style::Bold},
    {"__", Style::Bold},
}};

struct Block {
    enum class Kind {
        Paragraph,
        Heading,
        Quote,
        Code,
        Rule,
        ListItem,
    };

    Kind kind = Kind::Paragraph;
    std::string text;
    F32 fontSize = Typography::FontSize;
    F32 topGap = 0.0f;
    StyleId baseStyle = Style::Plain;
    F32 indent = 0.0f;
    std::string marker;

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
    };

    Kind kind = Kind::Code;
    U64 first = 0;
    U64 last = 0;
    std::string marker;
    F32 indent = 0.0f;
};

bool IsFence(const std::string& line) {
    return line.rfind("```", 0) == 0;
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

std::string TrimRight(std::string s) {
    while (!s.empty() && (s.back() == '\r' || s.back() == ' ' || s.back() == '\t')) {
        s.pop_back();
    }
    return s;
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
            if (scanFencedCode(line) || scanRule(line) || scanHeading(line) || scanQuote(line) || scanListItem(line)) {
                continue;
            }
            scanParagraph();
        }
        return std::move(blocks);
    }

    F32 gapAbove(F32 ratio) const {
        return blocks.empty() ? 0.0f : lineHeight * (prevHeading ? kAfterHeadingGapRatio : ratio);
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
        block.topGap = gapAbove(kParagraphGapRatio);
        block.baseStyle = Style::CodeBlock;
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
        block.topGap = gapAbove(kParagraphGapRatio);
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
        block.topGap = gapAbove(kHeadingGapRatio);
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
        block.kind = Block::Kind::Quote;
        block.text = JoinLines(quote);
        block.fontSize = body;
        block.topGap = gapAbove(kParagraphGapRatio);
        block.indent = body * kQuoteIndentEmRatio;
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
        block.topGap = lineHeight * (prevList ? kListGapRatio
                                   : prevHeading ? kAfterHeadingGapRatio
                                                 : kParagraphGapRatio);
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
            if (current.empty() || StartsBlock(current)) {
                break;
            }
            paragraph.push_back(current);
            ++index;
        }
        Block block;
        block.kind = Block::Kind::Paragraph;
        block.text = JoinLines(paragraph);
        block.fontSize = body;
        block.topGap = gapAbove(kParagraphGapRatio);
        emit(std::move(block));
    }
};

struct Document {
    std::vector<F32> lineScale;
    std::vector<F32> lineTopGap;
    std::vector<F32> lineIndent;
    std::vector<std::vector<StyleId>> styles;
    std::vector<std::vector<LinkSpan>> links;
    std::vector<Deco> decos;
    std::string plainValue;

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
    Document document;
    std::vector<std::string> lines;

    explicit DocumentBuilder(F32 body) : body(body) {}

    Document build(const std::vector<Block>& blocks) {
        for (const auto& block : blocks) {
            if (block.kind == Block::Kind::Rule) {
                emitRule(block);
            } else {
                emitTextBlock(block);
            }
        }
        if (lines.empty()) {
            addLine("", 1.0f, 0.0f, 0.0f, {}, {});
        }
        document.plainValue = JoinLines(lines);
        return std::move(document);
    }

    void addLine(std::string text, F32 scale, F32 gap, F32 indent,
                 std::vector<StyleId> lineStyles, std::vector<LinkSpan> lineLinks) {
        lines.push_back(std::move(text));
        document.lineScale.push_back(scale);
        document.lineTopGap.push_back(gap);
        document.lineIndent.push_back(indent);
        document.styles.push_back(std::move(lineStyles));
        document.links.push_back(std::move(lineLinks));
    }

    void emitRule(const Block& block) {
        const U64 line = lines.size();
        addLine("", kRuleRowScale, block.topGap, 0.0f, {}, {});
        document.decos.push_back({.kind = Deco::Kind::Rule, .first = line, .last = line});
    }

    void emitTextBlock(const Block& block) {
        const F32 scale = block.fontSize / body;
        const F32 indent = block.indent + (block.kind == Block::Kind::Code ? body * kCodePadRatio : 0.0f);
        const U64 firstLine = lines.size();

        const auto sources = SplitLines(block.text);
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
            }
            addLine(std::move(display), scale, gap, indent, std::move(styles), std::move(links));
        }
        const U64 lastLine = lines.size() - 1;

        switch (block.kind) {
            case Block::Kind::Code:
                document.decos.push_back({.kind = Deco::Kind::Code, .first = firstLine, .last = lastLine, .indent = indent});
                break;
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
        ColorRGBA<F32> bar;
        ColorRGBA<F32> rule;
        ColorRGBA<F32> marker;
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
        std::vector<Label::Instance> markers;

        bool full() const {
            return boxes.size() >= kMaxDecorations || markers.size() >= kMaxDecorations;
        }
    };

    Config config;
    TextGrid grid;
    Box decorations;
    Label markers;
    Document document;
    U64 styleRevision = 0;
    bool parsed = false;

    void rebuild() {
        ++styleRevision;
        const std::vector<Block> blocks = BlockScanner(SplitLines(config.value), config.fontSize).run();
        document = DocumentBuilder(config.fontSize).build(blocks);
    }

    F32 rowHeight(U64 line) const {
        return config.fontSize * document.scaleAt(line) * kLineHeightRatio;
    }

    TextGrid::Config buildGridConfig() {
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
            .padding = config.padding,
            .lineScale = document.lineScale,
            .lineTopGap = document.lineTopGap,
            .lineIndent = document.lineIndent,
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
            .styleScales = config.styleScales,
            .maxLineSegments = kGridLineSegments,
            .styleRevision = styleRevision,
            .styler = [this](const std::vector<std::string>&, U64)
                          -> const std::vector<std::vector<StyleId>>& {
                return document.styles;
            },
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
        const bool scrollbarVisible = config.scrollbar && metrics.contentHeight > rect.height;
        return {
            .rect = rect,
            .clip = clipPixel,
            .body = body,
            .bodyPad = metrics.padding.left,
            .width = std::max(0.0f, rect.width - (scrollbarVisible ? body * kScrollbarGutterFontRatio : 0.0f)),
            .on = !rect.empty(),
            .codeBackground = ctx.color(config.scrollbarTrackColorKey),
            .bar = ctx.color(config.lineNumberColorKey),
            .rule = ctx.color(config.gutterSeparatorColorKey),
            .marker = ctx.color(config.lineNumberColorKey),
            .metrics = &metrics,
        };
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

    void addCodeBackground(const DecorationFrame& frame, const Deco& deco, DecorationPools& pools) const {
        const F32 pad = frame.body * kCodePadRatio;
        const F32 top = frame.lineTop(deco.first);
        const F32 bottom = frame.lineBottom(deco.last);
        pools.boxes.push_back({
            .rect = {frame.rect.x, top - pad, frame.width, (bottom - top) + 2.0f * pad},
            .visible = frame.on,
            .backgroundColor = frame.codeBackground,
        });
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
        const F32 textStart = frame.rect.x + frame.bodyPad + deco.indent;
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

    DecorationPools buildDecorations(const DecorationFrame& frame) const {
        DecorationPools pools;
        for (const auto& deco : document.decos) {
            if (pools.full()) {
                break;
            }
            switch (deco.kind) {
                case Deco::Kind::Rule: addRule(frame, deco, pools); break;
                case Deco::Kind::Code: addCodeBackground(frame, deco, pools); break;
                case Deco::Kind::Quote: addQuoteBar(frame, deco, pools); break;
                case Deco::Kind::ListItem: addListMarker(frame, deco, pools); break;
            }
        }
        return pools;
    }

    void uploadDecorations(const DecorationFrame& frame, DecorationPools&& pools) {
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
    const Rect clipPixel = Intersect(bounds, clip());

    impl->grid.update(impl->buildGridConfig());
    layoutChild(ctx, impl->grid, bounds);

    const auto decoFrame = impl->decorationFrame(ctx, bounds, clipPixel);
    impl->uploadDecorations(decoFrame, impl->buildDecorations(decoFrame));

    layoutChild(ctx, impl->decorations, bounds);
    layoutChild(ctx, impl->markers, bounds);
}

}  // namespace Jetstream::Sakura::Retained
