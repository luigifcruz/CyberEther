#include <jetstream/render/sakura/components/terminal.hh>

#include <jetstream/render/sakura/clipboard.hh>
#include <jetstream/render/sakura/components/box.hh>
#include <jetstream/render/sakura/components/label.hh>
#include <jetstream/render/sakura/components/polyline.hh>
#include <jetstream/render/sakura/components/scroller.hh>

#include <jetstream/render/components/text.hh>

#include "../context.hh"
#include "../metrics.hh"
#include "../detail/terminal_sprites.hh"
#include "../detail/text_metrics.hh"

#include <algorithm>
#include <array>
#include <cmath>
#include <string>
#include <utility>
#include <vector>

namespace Jetstream::Sakura {

namespace {

using Unicode = Render::Components::Text::Unicode;

constexpr U16 kNaturalColumns = 80;
constexpr U16 kNaturalRows = 24;
constexpr F32 kMaxCells = 4096.0f;
constexpr U64 kGlyphGroupsPerRow = 4;
constexpr U64 kBoldGroupsPerRow = 2;
constexpr U64 kIsolatedGlyphsPerRow = 4;
constexpr U64 kSpriteRectsPerCell = 2;
constexpr U64 kSpriteStrokesPerRow = 16;
constexpr U64 kSpriteVerticesPerStroke = 48;
constexpr U64 kGraphemeCharacters = 16;
constexpr F32 kLineThicknessRatio = 0.05f;
constexpr std::array<const char*, 2> kFallbackFonts = {Typography::BodyFont, Typography::DisplayRegularFont};
constexpr U64 kRegularFace = 1;
constexpr U64 kFallbackFace = 2;
constexpr F32 kFallbackAdvanceRatio = 0.6f;
constexpr F32 kCursorBarWidth = 2.0f;
constexpr F32 kCursorUnderlineHeight = 2.0f;
constexpr F32 kWheelRows = 3.0f;
constexpr F64 kCursorBlinkPeriod = 1.0;
constexpr F64 kCursorBlinkOnDuration = 0.55;

}  // namespace

struct Terminal::Impl {
    struct Group {
        ColorRGBA<F32> color;
        U16 start = 0;
        U16 column = 0;
        std::string line;
    };

    struct Glyph {
        U64 face = 0;
        std::string text;
    };

    Terminal* self = nullptr;
    Config config;

    Box surface;
    Box backgrounds;
    Box cursorBox;
    Label glyphs;
    Label boldGlyphs;
    std::array<Label, kFallbackFonts.size()> fallbackGlyphs;
    Box sprites;
    Polyline spriteStrokes;
    Box decorations;
    Label cursorGlyph;
    Scroller scroller;

    TextMetrics metrics;
    TerminalSprites canvas;
    bool stale = true;
    F32 syncedRatio = 0.0f;
    Render::Window* syncedWindow = nullptr;
    F32 fontPixels = 0.0f;
    F32 cellWidth = 1.0f;
    F32 cellHeight = 1.0f;

    Rect grid;
    U16 columns = 1;
    U16 rows = 1;
    U64 rowCapacity = 0;
    U64 columnCapacity = 0;
    std::optional<Grid> reported;

    bool focused = false;
    bool selecting = false;
    bool pointing = false;
    std::optional<std::pair<U16, U16>> lastCell;
    F32 wheelRemainder = 0.0f;
    F64 now = 0.0;
    F64 blinkBase = 0.0;

    void sync(const Context& ctx) {
        if (!stale && syncedRatio == ctx.pixelRatio && syncedWindow == ctx.render) {
            return;
        }
        stale = false;
        syncedRatio = ctx.pixelRatio;
        syncedWindow = ctx.render;
        metrics.setWindow(ctx.render);
        fontPixels = std::max(1.0f, std::round(Px(ctx, config.fontSize)));
        const F32 measured = std::round(metrics.measure(Typography::MonoFont, "0", fontPixels));
        cellWidth = std::max(1.0f, measured > 0.0f ? measured : std::round(fontPixels * kFallbackAdvanceRatio));
        cellHeight = std::max(1.0f, std::round(fontPixels * Typography::CodeLineHeight));
        if (measured <= 0.0f) {
            syncedWindow = nullptr;
        }
    }

    Rect cellRect(U16 column, U16 row, U16 width = 1) const {
        return {grid.x + static_cast<F32>(column) * cellWidth,
                grid.y + static_cast<F32>(row) * cellHeight,
                static_cast<F32>(width) * cellWidth,
                cellHeight};
    }

    std::pair<U16, U16> cellAt(Extent2D<F32> point) const {
        const F32 x = std::floor((point.x - grid.x) / cellWidth);
        const F32 y = std::floor((point.y - grid.y) / cellHeight);
        return {static_cast<U16>(std::clamp(x, 0.0f, static_cast<F32>(columns - 1))),
                static_cast<U16>(std::clamp(y, 0.0f, static_cast<F32>(rows - 1)))};
    }

    bool inside(Extent2D<F32> point) const {
        return self->frame().contains(point.x, point.y);
    }

    bool blinkOn() const {
        return std::fmod(now - blinkBase, kCursorBlinkPeriod) < kCursorBlinkOnDuration;
    }

    F64 nextBlink() const {
        const F64 phase = std::fmod(now - blinkBase, kCursorBlinkPeriod);
        return now + (phase < kCursorBlinkOnDuration ? kCursorBlinkOnDuration - phase : kCursorBlinkPeriod - phase);
    }

    bool cursorBlinks() const {
        return focused && config.screen && config.screen->cursor.visible && config.screen->cursor.blinking;
    }

    void report() {
        const Grid next = {columns, rows, cellWidth, cellHeight};
        if (reported == next) {
            return;
        }
        reported = next;
        if (config.onResize) {
            config.onResize(next);
        }
    }

    void reserve(const Context& ctx) {
        const auto fit = [](U64 pixels, F32 cell) {
            return static_cast<U64>(std::ceil(static_cast<F32>(pixels) / cell)) + 1;
        };
        rowCapacity = std::max<U64>(rows + 1, fit(ctx.framebufferSize.y, cellHeight));
        columnCapacity = std::max<U64>(columns + 1, fit(ctx.framebufferSize.x, cellWidth));
    }

    std::optional<Glyph> resolve(const std::string& text, bool bold) {
        if (Unicode::IsAscii(text)) {
            return Glyph{.face = 0, .text = text};
        }
        std::vector<U32> codepoints;
        for (U64 position = 0; position < text.size();) {
            U32 codepoint = 0;
            position += Unicode::Decode(text, position, codepoint);
            codepoints.push_back(codepoint);
        }
        if (codepoints.empty()) {
            return std::nullopt;
        }
        for (U64 face = 0; face < kFallbackFace + kFallbackFonts.size(); ++face) {
            if (face == kRegularFace && !bold) {
                continue;
            }
            const std::string font = FontOf(face, bold);
            if (!metrics.covers(font, codepoints.front())) {
                continue;
            }
            Glyph glyph{.face = face};
            for (const U32 codepoint : codepoints) {
                if (metrics.covers(font, codepoint)) {
                    glyph.text += Unicode::Encode(codepoint);
                }
            }
            return glyph;
        }
        return std::nullopt;
    }

    static std::string FontOf(U64 face, bool bold) {
        if (face == 0) {
            return bold ? Typography::MonoBoldFont : Typography::MonoFont;
        }
        if (face == kRegularFace) {
            return Typography::MonoFont;
        }
        return kFallbackFonts[face - kFallbackFace];
    }

    static std::optional<U32> SpriteOf(const std::string& text) {
        if (text.empty() || Unicode::IsAscii(text)) {
            return std::nullopt;
        }
        U32 codepoint = 0;
        (void)Unicode::Decode(text, 0, codepoint);
        if (!TerminalSprites::Covers(codepoint)) {
            return std::nullopt;
        }
        return codepoint;
    }

    bool inlineGlyph(const Span& span, const std::string& font) {
        if (Unicode::IsAscii(span.text)) {
            return true;
        }
        const F32 advance = metrics.measure(font, span.text, fontPixels);
        return std::abs(advance - cellWidth) < 0.5f;
    }

    static Group& GroupFor(std::vector<Group>& groups, const ColorRGBA<F32>& color, U16 column) {
        for (auto& group : groups) {
            if (group.color == color) {
                return group;
            }
        }
        groups.push_back({.color = color, .start = column, .column = column});
        return groups.back();
    }

    static void Append(Group& group, const Span& span) {
        if (span.column > group.column) {
            group.line.append(span.column - group.column, ' ');
        }
        group.line += span.text;
        const U16 cells = static_cast<U16>(Unicode::IsAscii(span.text) ? span.text.size() : 1);
        if (span.width > cells) {
            group.line.append(span.width - cells, ' ');
        }
        group.column = std::max<U16>(group.column, span.column + std::max<U16>(span.width, cells));
    }

    void build(const Context& ctx) {
        const Screen* screen = config.screen;
        const ColorRGBA<F32> cursorColor = ctx.color(config.cursorColorKey);
        const ColorRGBA<F32> selectionColor = ctx.color(config.selectionColorKey);
        const ColorRGBA<F32> backdropColor =
            ctx.color(ctx.backdropColorKey.empty() ? "background" : ctx.backdropColorKey);

        std::vector<Box::Instance> fills;
        std::vector<Label::Instance> regular;
        std::vector<Label::Instance> bold;
        std::vector<Box::Instance> lines;
        std::vector<Label::Instance> isolated;
        std::array<std::vector<Label::Instance>, kFallbackFonts.size()> fallback;
        const U64 cells = rowCapacity * columnCapacity;
        const U64 fillCapacity = cells + rowCapacity;
        const U64 glyphCapacity = rowCapacity * (kGlyphGroupsPerRow + kIsolatedGlyphsPerRow);
        const U64 boldCapacity = rowCapacity * kBoldGroupsPerRow;
        const U64 fallbackCapacity = rowCapacity * kIsolatedGlyphsPerRow;
        const U64 lineCapacity = cells;
        const U64 spriteCapacity = cells * kSpriteRectsPerCell;
        const U64 strokeCapacity = rowCapacity * kSpriteStrokesPerRow;
        const F32 hairline = Hairline(ctx);
        const F32 lineThickness = std::max(1.0f, std::ceil(fontPixels * kLineThicknessRatio));
        canvas.clear();
        canvas.rectCapacity = spriteCapacity;
        canvas.strokeCapacity = strokeCapacity;

        const auto isolate = [&](const Glyph& glyph, const Rect& rect, const ColorRGBA<F32>& color) {
            Label::Instance instance{
                .rect = rect,
                .str = glyph.text,
                .color = color,
                .fontSize = fontPixels,
                .alignment = {1, 1},
            };
            if (glyph.face >= kFallbackFace) {
                auto& target = fallback[glyph.face - kFallbackFace];
                if (target.size() < fallbackCapacity) {
                    target.push_back(std::move(instance));
                }
                return;
            }
            if (regular.size() + isolated.size() < glyphCapacity) {
                isolated.push_back(std::move(instance));
            }
        };

        const U16 visibleRows = screen ? std::min<U16>(rows, static_cast<U16>(screen->lines.size())) : 0;
        const bool cursorCell = screen && screen->cursor.visible && screen->cursor.row < visibleRows;
        const bool tinted = screen && screen->fill.a > 0.0f && screen->fill != screen->base;
        const ColorRGBA<F32> under = screen ? (tinted ? screen->fill : screen->base) : ColorRGBA<F32>{};
        const bool defaults = screen && screen->background.a > 0.0f && screen->background != under;
        surface.update({
            .instances = {{
                .rect = self->frame(),
                .visible = tinted,
                .backgroundColor = tinted ? screen->fill : ColorRGBA<F32>{0.0f, 0.0f, 0.0f, 0.0f},
                .corners = config.corners,
            }},
            .cornerRadius = Px(ctx, config.rounding),
            .capacity = 1,
        });
        const bool cursorShown = cursorCell && (!cursorBlinks() || blinkOn());
        const bool solidCursor = cursorShown && focused && screen->cursor.shape == CursorShape::Block;

        std::vector<Group> groups;
        std::vector<Group> boldGroups;
        for (U16 row = 0; row < visibleRows; ++row) {
            const Row& line = screen->lines[row];
            const auto shade = [&](U16 column, U16 end, const ColorRGBA<F32>& color) {
                end = std::min(end, columns);
                if (column < end && fills.size() < fillCapacity && (under.a == 0.0f || color != under)) {
                    fills.push_back({.rect = cellRect(column, row, end - column), .backgroundColor = color});
                }
            };
            U16 painted = 0;
            for (const auto& run : line.backgrounds) {
                if (defaults && run.column > painted) {
                    shade(painted, run.column, screen->background);
                }
                shade(run.column, run.column + run.width, run.color);
                painted = std::max<U16>(painted, run.column + run.width);
            }
            if (defaults && painted < columns) {
                shade(painted, columns, screen->background);
            }
            if (line.selection.has_value() && fills.size() < fillCapacity && line.selection->start < columns) {
                const U16 end = std::min<U16>(line.selection->end, columns - 1);
                if (end >= line.selection->start) {
                    fills.push_back({
                        .rect = cellRect(line.selection->start, row, end - line.selection->start + 1),
                        .backgroundColor = selectionColor,
                    });
                }
            }

            groups.clear();
            boldGroups.clear();
            for (const auto& span : line.spans) {
                if (span.column >= columns || span.text.empty()) {
                    continue;
                }
                const U16 width = std::min<U16>(std::max<U16>(1, span.width), columns - span.column);
                if (const auto sprite = SpriteOf(span.text)) {
                    canvas.draw(*sprite, cellRect(span.column, row, width), span.color, lineThickness);
                } else if (const auto glyph = resolve(span.text, span.bold); glyph && !glyph->text.empty()) {
                    Span shown = span;
                    shown.text = glyph->text;
                    if (glyph->face == 0 && inlineGlyph(shown, FontOf(glyph->face, span.bold))) {
                        Append(GroupFor(span.bold ? boldGroups : groups, span.color, span.column), shown);
                    } else {
                        isolate(*glyph, cellRect(span.column, row, width), span.color);
                    }
                }
                if ((span.underline || span.strikethrough) && lines.size() < lineCapacity) {
                    const Rect cell = cellRect(span.column, row, width);
                    const F32 y = span.underline ? cell.bottom() - hairline : std::round(cell.y + cell.height * 0.5f);
                    lines.push_back({.rect = {cell.x, y, cell.width, hairline}, .backgroundColor = span.color});
                }
            }
            const auto emit = [&](const std::vector<Group>& source, std::vector<Label::Instance>& target, U64 capacity) {
                for (const auto& group : source) {
                    if (target.size() >= capacity) {
                        return;
                    }
                    const U16 span = static_cast<U16>(std::max<U16>(1, group.column - group.start));
                    target.push_back({
                        .rect = cellRect(group.start, row, span),
                        .str = group.line,
                        .color = group.color,
                        .fontSize = fontPixels,
                        .alignment = {0, 1},
                    });
                }
            };
            emit(groups, regular, glyphCapacity - std::min<U64>(glyphCapacity, isolated.size()));
            emit(boldGroups, bold, boldCapacity);
        }

        Box::Instance cursor{.visible = false};
        Label::Instance cursorText{.visible = false};
        bool hollow = false;
        if (cursorCell) {
            const auto& state = screen->cursor;
            const Rect cell = cellRect(state.column, state.row, std::max<U16>(1, state.width));
            if (cursorShown) {
                const CursorShape shape = focused ? state.shape : CursorShape::Hollow;
                Rect rect = cell;
                if (shape == CursorShape::Bar) {
                    rect.width = std::max(hairline, std::round(Px(ctx, kCursorBarWidth)));
                } else if (shape == CursorShape::Underline) {
                    const F32 height = std::max(hairline, std::round(Px(ctx, kCursorUnderlineHeight)));
                    rect = {cell.x, cell.bottom() - height, cell.width, height};
                }
                hollow = shape == CursorShape::Hollow;
                cursor = {
                    .rect = rect,
                    .visible = true,
                    .backgroundColor = hollow ? ColorRGBA<F32>{0.0f, 0.0f, 0.0f, 0.0f} : cursorColor,
                };
            }
            const ColorRGBA<F32> behind = tinted ? screen->fill : backdropColor;
            const ColorRGBA<F32> textColor =
                solidCursor ? ColorRGBA<F32>{behind.r, behind.g, behind.b, 1.0f} : state.textColor;
            if (const auto sprite = SpriteOf(state.text)) {
                canvas.draw(*sprite, cell, textColor, lineThickness);
            } else if (const auto glyph = resolve(state.text, false); glyph && !glyph->text.empty()) {
                if (glyph->face == 0) {
                    cursorText = {
                        .rect = cell,
                        .str = glyph->text,
                        .visible = true,
                        .color = textColor,
                        .fontSize = fontPixels,
                        .alignment = {Unicode::IsAscii(glyph->text) ? 0 : 1, 1},
                    };
                } else {
                    isolate(*glyph, cell, textColor);
                }
            }
        }
        regular.insert(regular.end(), isolated.begin(), isolated.end());

        backgrounds.update({.instances = std::move(fills), .capacity = fillCapacity});
        cursorBox.update({
            .instances = {cursor},
            .borderWidth = hollow ? hairline : 0.0f,
            .borderColor = cursorColor,
        });
        glyphs.update({
            .instances = std::move(regular),
            .fontName = Typography::MonoFont,
            .maxCharacters = columnCapacity,
            .capacity = glyphCapacity,
        });
        boldGlyphs.update({
            .instances = std::move(bold),
            .fontName = Typography::MonoBoldFont,
            .maxCharacters = columnCapacity,
            .capacity = boldCapacity,
        });
        for (U64 index = 0; index < fallbackGlyphs.size(); ++index) {
            fallbackGlyphs[index].update({
                .instances = std::move(fallback[index]),
                .fontName = kFallbackFonts[index],
                .maxCharacters = kGraphemeCharacters,
                .capacity = fallbackCapacity,
            });
        }
        sprites.update({.instances = canvas.rects, .capacity = spriteCapacity});
        spriteStrokes.update({
            .instances = canvas.strokes,
            .fringe = Fringe(ctx),
            .capacity = strokeCapacity * kSpriteVerticesPerStroke,
        });
        decorations.update({.instances = std::move(lines), .capacity = lineCapacity});
        cursorGlyph.update({
            .instances = {cursorText},
            .fontName = Typography::MonoFont,
            .maxCharacters = kGraphemeCharacters,
            .capacity = 1,
        });
    }

    void configureScroller(const Context& ctx) {
        const Screen* screen = config.screen;
        const U64 visible = screen ? std::max<U64>(1, screen->rows) : 1;
        const U64 total = screen ? std::max(screen->scrollTotal, visible) : visible;
        const U64 offset = screen ? std::min(screen->scrollOffset, total - visible) : 0;
        scroller.update({
            .id = config.id + ":scroll",
            .contentHeight = self->frame().height + static_cast<F32>(total - visible) * cellHeight,
            .scrollY = static_cast<F32>(offset) * cellHeight,
            .scrollbar = true,
            .wheel = false,
            .wheelStep = Px(ctx, kWheelStep),
            .thickness = Px(ctx, kScrollbarThickness),
            .margin = Px(ctx, kScrollbarMargin),
            .trackColorKey = config.scrollbarTrackColorKey,
            .thumbColorKey = config.scrollbarThumbColorKey,
            .onScrollY = [this](F32 scrollY) {
                if (config.onScrollTo) {
                    config.onScrollTo(static_cast<U64>(std::max(0.0f, std::round(scrollY / cellHeight))));
                }
            },
        });
    }

    void copy() {
        if (config.screen && !config.screen->selection.empty()) {
            SetClipboardText(metrics.window, config.screen->selection);
        }
    }

    void paste() {
        const std::string text = GetClipboardText(metrics.window);
        if (!text.empty() && config.onPaste) {
            config.onPaste(text);
        }
    }

    MouseEvent local(const MouseEvent& event) const {
        MouseEvent shifted = event;
        shifted.position = {event.position.x - grid.x, event.position.y - grid.y};
        return shifted;
    }

    void point(const MouseEvent& event) {
        if (config.onPointer) {
            config.onPointer(local(event));
        }
    }

    void select(Gesture gesture, Extent2D<F32> point, U8 clicks) {
        const auto [column, row] = cellAt(point);
        if (gesture == Gesture::Drag && lastCell == std::make_pair(column, row)) {
            return;
        }
        lastCell = {column, row};
        if (config.onSelect) {
            config.onSelect({.gesture = gesture, .column = column, .row = row, .clicks = clicks});
        }
    }
};

Terminal::Terminal() {
    impl = std::make_unique<Impl>();
    impl->self = this;
    setClipsChildren(true);
    addChild(impl->surface);
    addChild(impl->backgrounds);
    addChild(impl->cursorBox);
    addChild(impl->glyphs);
    addChild(impl->boldGlyphs);
    for (auto& label : impl->fallbackGlyphs) {
        addChild(label);
    }
    addChild(impl->sprites);
    addChild(impl->spriteStrokes);
    addChild(impl->decorations);
    addChild(impl->cursorGlyph);
    addChild(impl->scroller);
}

Terminal::~Terminal() = default;

void Terminal::update(Config config) {
    if (config != impl->config) {
        invalidate(Dirty::Layout);
    }
    const bool requested = config.screen && config.focusRequest != impl->config.focusRequest;
    const bool attached = config.screen && (config.screen != impl->config.screen || requested);
    if (config.screen != impl->config.screen) {
        impl->reported.reset();
    }
    impl->stale = impl->stale || config.fontSize != impl->config.fontSize;
    impl->config = std::move(config);
    if (requested && textEditorFocused()) {
        requestFocus();
    }
    if (attached) {
        offerFocus(true);
    }
}

Extent2D<F32> Terminal::measure(const Context& ctx, Extent2D<F32> available) {
    impl->sync(ctx);
    const F32 padding = 2.0f * Px(ctx, impl->config.padding);
    const Extent2D<F32> natural = {
        static_cast<F32>(kNaturalColumns) * impl->cellWidth + padding,
        static_cast<F32>(kNaturalRows) * impl->cellHeight + padding,
    };
    return {ResolveLength(ctx, impl->config.size.x, natural.x, available.x),
            ResolveLength(ctx, impl->config.size.y, natural.y, available.y)};
}

void Terminal::layout(const Context& ctx) {
    impl->sync(ctx);
    impl->now = ctx.time;
    const Rect bounds = frame();
    const F32 padding = Px(ctx, impl->config.padding);
    impl->grid = {
        std::round(bounds.x + padding),
        std::round(bounds.y + padding),
        std::max(0.0f, bounds.width - 2.0f * padding),
        std::max(0.0f, bounds.height - 2.0f * padding),
    };
    impl->columns = static_cast<U16>(std::clamp(std::floor(impl->grid.width / impl->cellWidth), 1.0f, kMaxCells));
    impl->rows = static_cast<U16>(std::clamp(std::floor(impl->grid.height / impl->cellHeight), 1.0f, kMaxCells));
    impl->reserve(ctx);
    if (!bounds.empty()) {
        impl->report();
    }

    impl->configureScroller(ctx);
    layoutChild(ctx, impl->scroller, bounds);
    impl->build(ctx);

    const Rect content = impl->grid;
    layoutChild(ctx, impl->surface, bounds);
    layoutChild(ctx, impl->backgrounds, content);
    layoutChild(ctx, impl->cursorBox, content);
    layoutChild(ctx, impl->glyphs, content);
    layoutChild(ctx, impl->boldGlyphs, content);
    for (auto& label : impl->fallbackGlyphs) {
        layoutChild(ctx, label, content);
    }
    layoutChild(ctx, impl->sprites, content);
    layoutChild(ctx, impl->spriteStrokes, content);
    layoutChild(ctx, impl->decorations, content);
    layoutChild(ctx, impl->cursorGlyph, content);

    if (impl->cursorBlinks()) {
        requestFrame(impl->nextBlink());
    }
}

bool Terminal::mouseEvent(const MouseEvent& event) {
    if (!impl->selecting && !impl->pointing && mouseEventChildren(event)) {
        return true;
    }
    const Screen* screen = impl->config.screen;
    const bool tracking = screen && screen->mouseTracking;
    switch (event.type) {
        case MouseEventType::Click: {
            if (!impl->inside(event.position)) {
                return false;
            }
            if (tracking && !event.modifiers.shift) {
                impl->pointing = true;
                impl->lastCell = impl->cellAt(event.position);
                impl->point(event);
                return true;
            }
            if (event.button != MouseButton::Left) {
                return false;
            }
            impl->selecting = true;
            impl->lastCell.reset();
            impl->select(Gesture::Press, event.position, event.clicks);
            return true;
        }
        case MouseEventType::Move: {
            if (impl->selecting) {
                requestCursor(SurfaceCursor::TextInput);
                impl->select(Gesture::Drag, event.position, 1);
                return true;
            }
            if (impl->pointing || (tracking && impl->inside(event.position))) {
                const auto cell = impl->cellAt(event.position);
                if (cell != impl->lastCell) {
                    impl->lastCell = cell;
                    impl->point(event);
                }
                return impl->pointing;
            }
            return false;
        }
        case MouseEventType::Release: {
            if (impl->selecting) {
                impl->selecting = false;
                impl->select(Gesture::Release, event.position, 1);
                return true;
            }
            if (impl->pointing) {
                impl->pointing = false;
                impl->point(event);
                return true;
            }
            return false;
        }
        case MouseEventType::Scroll: {
            if (!impl->inside(event.position) || event.scroll.y == 0.0f) {
                return false;
            }
            const F32 unit = WheelStep(event, kWheelRows * impl->cellHeight, impl->syncedRatio);
            impl->wheelRemainder -= event.scroll.y * unit / impl->cellHeight;
            const F32 whole = std::trunc(impl->wheelRemainder);
            if (whole != 0.0f) {
                impl->wheelRemainder -= whole;
                if (impl->config.onScroll) {
                    impl->config.onScroll(static_cast<I32>(whole), impl->local(event).position, event.modifiers);
                }
            }
            return true;
        }
        case MouseEventType::Cancel:
        case MouseEventType::Leave: {
            if (impl->pointing) {
                MouseEvent cancel = event;
                cancel.type = MouseEventType::Cancel;
                impl->point(cancel);
            }
            impl->selecting = false;
            impl->pointing = false;
            impl->lastCell.reset();
            return false;
        }
        default:
            return false;
    }
}

bool Terminal::keyEvent(const KeyEvent& event) {
    if (event.type != KeyEventType::Press) {
        return false;
    }
    const auto& modifiers = event.modifiers;
    const bool shiftControl = modifiers.control && modifiers.shift && !modifiers.alt && !modifiers.super;
    if ((modifiers.super && !modifiers.control && !modifiers.alt) || shiftControl) {
        if (event.key == KeyCode::C) {
            if (!event.repeat) {
                impl->copy();
            }
            return true;
        }
        if (event.key == KeyCode::V) {
            if (!event.repeat) {
                impl->paste();
            }
            return true;
        }
    }
    if (modifiers.super || (modifiers.control && event.key == KeyCode::GraveAccent)) {
        return false;
    }
    if (impl->config.onKey) {
        impl->config.onKey(event);
    }
    impl->blinkBase = impl->now;
    invalidate(Dirty::Layout);
    return true;
}

bool Terminal::textEvent(const TextEvent& event) {
    const char32_t codepoint = event.codepoint;
    const auto& modifiers = event.modifiers;
    if (codepoint < 32 || codepoint == 127 || modifiers.super || (modifiers.control && !modifiers.alt) ||
        (impl->config.altSendsEscape && modifiers.alt && !modifiers.control)) {
        return true;
    }
    if (impl->config.onText) {
        impl->config.onText(Unicode::Encode(static_cast<U32>(codepoint)));
    }
    return true;
}

void Terminal::focusChanged(bool focused) {
    impl->focused = focused;
    impl->blinkBase = impl->now;
    invalidate(Dirty::Layout);
    if (impl->config.onFocusChange) {
        impl->config.onFocusChange(focused);
    }
}

bool Terminal::editsText() const {
    return true;
}

bool Terminal::focusesOnClick() const {
    return true;
}

std::optional<SurfaceCursor> Terminal::cursorAt(const Extent2D<F32>& point) const {
    if (!impl->inside(point)) {
        return std::nullopt;
    }
    return SurfaceCursor::TextInput;
}

}  // namespace Jetstream::Sakura
