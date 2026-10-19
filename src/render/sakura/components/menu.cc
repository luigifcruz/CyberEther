#include <jetstream/render/sakura/components/menu.hh>

#include <jetstream/render/sakura/components/box.hh>
#include <jetstream/render/sakura/components/polyline.hh>
#include <jetstream/render/sakura/components/text.hh>
#include <jetstream/render/sakura/typography.hh>

#include "../context.hh"
#include "../metrics.hh"
#include "../detail/helpers.hh"
#include "../detail/popup_layer.hh"

#include <algorithm>
#include <cmath>
#include <limits>
#include <optional>
#include <utility>

namespace Jetstream::Sakura {

namespace {

bool SameEntries(const std::vector<MenuEntry>& a, const std::vector<MenuEntry>& b) {
    if (a.size() != b.size()) {
        return false;
    }
    for (U64 index = 0; index < a.size(); ++index) {
        const auto& left = a[index];
        const auto& right = b[index];
        if (left.label != right.label || left.shortcut != right.shortcut || left.selected != right.selected ||
            left.enabled != right.enabled || left.separator != right.separator || left.colorKey != right.colorKey ||
            !SameEntries(left.children, right.children)) {
            return false;
        }
    }
    return true;
}

std::vector<Extent2D<F32>> CheckMark(Extent2D<F32> pos, F32 size, F32& thickness) {
    thickness = std::max(size / 5.0f, 1.0f);
    size -= thickness * 0.5f;
    pos.x += thickness * 0.25f;
    pos.y += thickness * 0.25f;
    const F32 third = size / 3.0f;
    const F32 bx = pos.x + third;
    const F32 by = pos.y + size - third * 0.5f;
    return {{bx - third, by - third}, {bx, by}, {bx + third * 2.0f, by - third * 2.0f}};
}

std::vector<Extent2D<F32>> RightArrow(Extent2D<F32> pos, F32 size) {
    const F32 r = size * 0.40f;
    const Extent2D<F32> center = {pos.x + size * 0.5f, pos.y + size * 0.5f};
    return {
        {center.x + 0.750f * r, center.y},
        {center.x - 0.750f * r, center.y + 0.866f * r},
        {center.x - 0.750f * r, center.y - 0.866f * r},
    };
}

struct MenuShared {
    std::function<void()> close;
};

}  // namespace

struct MenuBody : public Component {
    struct Row {
        Text label;
        Text shortcut;
    };

    Box panel;
    Box highlights;
    Box separators;
    Polyline marks;
    std::vector<std::unique_ptr<Row>> rows;
    std::vector<MenuEntry> entries;
    std::vector<Rect> rowRects;
    MenuShared* shared = nullptr;
    std::optional<U64> hovered;
    std::optional<U64> pressed;
    std::optional<U64> openSubmenu;
    std::optional<U64> childIndex;
    std::vector<Component*> mounted;
    std::optional<U64> sourceIndex;
    U64 depth = 0;
    std::function<MenuBody*(U64)> level;

    MenuBody() {
        setClipsChildren(true);
    }

    void clear() {
        openSubmenu.reset();
        childIndex.reset();
        sourceIndex.reset();
        hovered.reset();
        pressed.reset();
        invalidate(Dirty::Layout);
    }

    void setEntries(const std::vector<MenuEntry>& next) {
        const bool changed = !SameEntries(entries, next);
        entries = next;
        while (rows.size() < entries.size()) {
            rows.push_back(std::make_unique<Row>());
        }
        std::vector<Component*> children = {&panel, &highlights, &separators};
        for (U64 index = 0; index < entries.size(); ++index) {
            const auto& entry = entries[index];
            if (entry.separator) {
                continue;
            }
            auto& row = *rows[index];
            const F32 opacity = entry.enabled ? 1.0f : kDisabledAlpha;
            row.label.update({
                .str = entry.label,
                .colorKey = entry.colorKey.empty() ? "text_primary" : entry.colorKey,
                .opacity = opacity,
            });
            row.shortcut.update({
                .str = entry.shortcut,
                .colorKey = "text_secondary",
                .opacity = opacity,
            });
            children.push_back(&row.label);
            if (!entry.shortcut.empty()) {
                children.push_back(&row.shortcut);
            }
        }
        children.push_back(&marks);
        if (children != mounted) {
            mountChildren(children);
            mounted = std::move(children);
        }
        if (openSubmenu.has_value() && *openSubmenu >= entries.size()) {
            openSubmenu.reset();
        }
        if (changed) {
            invalidate(Dirty::Layout);
        }
    }

    bool interactive(U64 index) const {
        return index < entries.size() && !entries[index].separator && entries[index].enabled;
    }

    bool submenu(U64 index) const {
        return interactive(index) && !entries[index].children.empty();
    }

    struct Columns {
        F32 label = 0.0f;
        F32 shortcut = 0.0f;
        F32 mark = 0.0f;
    };

    Columns columns(const Context& ctx) {
        const F32 infinite = std::numeric_limits<F32>::infinity();
        Columns result;
        for (U64 index = 0; index < entries.size(); ++index) {
            if (entries[index].separator) {
                continue;
            }
            result.label = std::max(result.label, measureChild(rows[index]->label, ctx, {infinite, infinite}).x);
            if (!entries[index].shortcut.empty()) {
                result.shortcut = std::max(result.shortcut,
                                           measureChild(rows[index]->shortcut, ctx, {infinite, infinite}).x);
            }
        }
        result.mark = std::floor(Px(ctx, Typography::FontSize) * 1.20f);
        return result;
    }

    F32 contentWidth(const Context& ctx, const Columns& cols) const {
        const F32 spacing = Px(ctx, kItemSpacing);
        F32 width = cols.label;
        if (cols.shortcut > 0.0f) {
            width += spacing + cols.shortcut;
        }
        return width + spacing + cols.mark;
    }

    F32 rowHeight(const Context& ctx, U64 index) {
        if (entries[index].separator) {
            return Hairline(ctx);
        }
        const F32 infinite = std::numeric_limits<F32>::infinity();
        return measureChild(rows[index]->label, ctx, {infinite, infinite}).y;
    }

    F32 contentHeight(const Context& ctx) {
        const F32 spacing = Px(ctx, kItemSpacing);
        F32 height = 0.0f;
        for (U64 index = 0; index < entries.size(); ++index) {
            height += rowHeight(ctx, index);
            if (index + 1 < entries.size()) {
                height += spacing;
            }
        }
        return height;
    }

    Extent2D<F32> measure(const Context& ctx, Extent2D<F32> available) override {
        const F32 padding = Px(ctx, kWindowPadding);
        const auto cols = columns(ctx);
        return FitExtent({std::ceil(contentWidth(ctx, cols) + 2.0f * padding),
                          std::ceil(contentHeight(ctx) + 2.0f * padding)}, available);
    }

    void layout(const Context& ctx) override {
        const Rect bounds = frame();
        const F32 padding = Px(ctx, kWindowPadding);
        const F32 spacing = Px(ctx, kItemSpacing);
        const F32 half = std::floor(spacing * 0.5f);
        const F32 font = Px(ctx, Typography::FontSize);
        const F32 line = Hairline(ctx);
        const auto cols = columns(ctx);
        const F32 markOffset = cols.label + spacing + (cols.shortcut > 0.0f ? cols.shortcut + spacing : 0.0f);
        const F32 shortcutOffset = cols.label + spacing;

        panel.update({
            .instances = {{.rect = bounds, .visible = !bounds.empty(), .backgroundColor = ctx.color("popup_bg")}},
            .cornerRadius = Px(ctx, kPopupRounding),
            .borderWidth = Px(ctx, 1.0f),
            .borderColor = ctx.color("border"),
            .strokeFringe = true,
        });
        layoutChild(ctx, panel, bounds);

        std::vector<Box::Instance> highlightInstances;
        std::vector<Box::Instance> separatorInstances;
        std::vector<Polyline::Instance> markInstances;
        rowRects.assign(entries.size(), Rect{});
        F32 y = bounds.y + padding;
        const F32 x = bounds.x + padding;
        childIndex = openSubmenu.has_value() && submenu(*openSubmenu) ? openSubmenu : std::nullopt;
        for (U64 index = 0; index < entries.size(); ++index) {
            const auto& entry = entries[index];
            if (entry.separator) {
                separatorInstances.push_back({
                    .rect = {bounds.x + line, std::floor(y), std::max(0.0f, bounds.width - 2.0f * line), line},
                    .backgroundColor = ctx.color("separator"),
                });
                y += line + spacing;
                continue;
            }
            const F32 height = rowHeight(ctx, index);
            const Rect highlight = {x - half, y - half, std::max(0.0f, bounds.width - 2.0f * padding) + spacing,
                                    height + spacing};
            rowRects[index] = highlight;
            const bool open = childIndex == index;
            const bool isHovered = hovered == index && entry.enabled;
            if (isHovered || open) {
                const char* key = isHovered && pressed == index ? "header_active" : isHovered ? "header_hovered" : "header";
                highlightInstances.push_back({.rect = highlight, .backgroundColor = ctx.color(key)});
            }
            auto& row = *rows[index];
            layoutChild(ctx, row.label, {x, y, cols.label, height});
            if (!entry.shortcut.empty()) {
                layoutChild(ctx, row.shortcut, {x + shortcutOffset, y, cols.shortcut, height});
            }
            auto color = ctx.color(entry.colorKey.empty() ? "text_primary" : entry.colorKey);
            color = WithAlpha(color, entry.enabled ? 1.0f : kDisabledAlpha);
            if (!entry.children.empty()) {
                markInstances.push_back({
                    .points = RightArrow({x + markOffset + font * 0.30f, y}, font),
                    .color = color,
                    .closed = true,
                    .filled = true,
                });
            } else if (entry.selected) {
                F32 thickness = 1.0f;
                auto points = CheckMark({x + markOffset + font * 0.40f, y + font * 0.134f * 0.5f}, font * 0.866f,
                                        thickness);
                markInstances.push_back({.points = std::move(points), .color = color, .thickness = thickness});
            }
            y += height + spacing;
        }
        highlights.update({
            .instances = std::move(highlightInstances),
            .clip = bounds,
            .cornerRadius = Px(ctx, kFrameRounding),
        });
        separators.update({.instances = std::move(separatorInstances), .clip = bounds});
        marks.update({.instances = std::move(markInstances), .clip = bounds, .fringe = Fringe(ctx), .capacity = 256});
        layoutChild(ctx, highlights, bounds);
        layoutChild(ctx, separators, bounds);
        layoutChild(ctx, marks, bounds);

        if (childIndex.has_value() && *childIndex < rowRects.size() && level) {
            if (auto* child = level(depth + 1); child && child->sourceIndex == childIndex) {
                RequestPopup(ctx, {
                    .content = child,
                    .anchor = rowRects[*childIndex],
                    .placement = PopupPlacement::Side,
                    .onDismiss = shared ? shared->close : nullptr,
                });
            }
        }
    }

    std::optional<U64> rowAt(const Extent2D<F32>& point) const {
        for (U64 index = 0; index < rowRects.size() && index < entries.size(); ++index) {
            if (!entries[index].separator && rowRects[index].contains(point.x, point.y)) {
                return index;
            }
        }
        return std::nullopt;
    }

    void openChild(std::optional<U64> next) {
        openSubmenu = next;
        if (!next.has_value() || !level || !submenu(*next)) {
            return;
        }
        auto* child = level(depth + 1);
        if (!child || child->sourceIndex == next) {
            return;
        }
        child->clear();
        child->sourceIndex = next;
        child->setEntries(entries[*next].children);
    }

    bool mouseEvent(const MouseEvent& event) override {
        const auto row = rowAt(event.position);
        switch (event.type) {
            case MouseEventType::Move: {
                if (row != hovered) {
                    hovered = row;
                    invalidate(Dirty::Layout);
                }
                if (row.has_value()) {
                    const std::optional<U64> next = submenu(*row) ? row : std::nullopt;
                    if (next != openSubmenu) {
                        openChild(next);
                        invalidate(Dirty::Layout);
                    }
                }
                return row.has_value();
            }
            case MouseEventType::Leave:
            case MouseEventType::Cancel:
                if (hovered.has_value() || pressed.has_value()) {
                    hovered.reset();
                    pressed.reset();
                    invalidate(Dirty::Layout);
                }
                return false;
            case MouseEventType::Click:
                if (event.button != MouseButton::Left || !row.has_value()) {
                    return row.has_value();
                }
                if (submenu(*row)) {
                    openChild(row);
                } else if (interactive(*row)) {
                    pressed = row;
                }
                invalidate(Dirty::Layout);
                return true;
            case MouseEventType::Release: {
                if (event.button != MouseButton::Left || !pressed.has_value()) {
                    return false;
                }
                pressed.reset();
                invalidate(Dirty::Layout);
                if (row.has_value() && interactive(*row) && !submenu(*row)) {
                    const auto onClick = entries[*row].onClick;
                    if (shared && shared->close) {
                        shared->close();
                    }
                    if (onClick) {
                        onClick();
                    }
                }
                return true;
            }
            default:
                return false;
        }
    }

    std::optional<SurfaceCursor> cursorAt(const Extent2D<F32>& point) const override {
        if (!frame().contains(point.x, point.y)) {
            return std::nullopt;
        }
        return SurfaceCursor::Default;
    }
};

struct Menu::Impl {
    Config config;
    MenuShared shared;
    std::vector<std::unique_ptr<MenuBody>> levels;
    bool wasOpen = false;
    Extent2D<F32> cursor = {0.0f, 0.0f};

    MenuBody* level(U64 depth) {
        while (levels.size() <= depth) {
            auto body = std::make_unique<MenuBody>();
            body->depth = levels.size();
            body->shared = &shared;
            body->level = [this](U64 next) {
                return level(next);
            };
            levels.push_back(std::move(body));
        }
        return levels[depth].get();
    }

    void reset() {
        for (auto& body : levels) {
            body->clear();
        }
    }
};

Menu::Menu() {
    impl = std::make_unique<Impl>();
    impl->shared.close = [this]() {
        if (!impl->config.open) {
            return;
        }
        impl->config.open = false;
        invalidate(Dirty::Layout);
        if (impl->config.onClose) {
            impl->config.onClose();
        }
    };
}

Menu::~Menu() = default;

void Menu::update(Config config) {
    if (config != impl->config) {
        invalidate(Dirty::Layout);
    }
    impl->config = std::move(config);
    auto* root = impl->level(0);
    root->setEntries(impl->config.entries);
    for (U64 depth = 0; depth + 1 < impl->levels.size(); ++depth) {
        auto* body = impl->levels[depth].get();
        auto* child = impl->levels[depth + 1].get();
        if (body->sourceIndex.has_value() || depth == 0) {
            if (child->sourceIndex.has_value() && *child->sourceIndex < body->entries.size()) {
                child->setEntries(body->entries[*child->sourceIndex].children);
            }
        }
    }
}

Extent2D<F32> Menu::measure(const Context&, Extent2D<F32>) {
    return {0.0f, 0.0f};
}

void Menu::layout(const Context& ctx) {
    const auto& config = impl->config;
    if (config.open && !impl->wasOpen) {
        impl->reset();
        impl->cursor = ctx.mouse.value_or(Extent2D<F32>{frame().x, frame().y});
    }
    impl->wasOpen = config.open;
    if (!config.open) {
        return;
    }
    Rect anchor = config.atCursor ? Rect{impl->cursor.x, impl->cursor.y, 1.0f, 1.0f} : frame();
    if (config.point.has_value()) {
        anchor = {config.point->x, config.point->y, 1.0f, 1.0f};
    }
    RequestPopup(ctx, {
        .content = impl->level(0),
        .anchor = anchor,
        .placement = config.placement,
        .onDismiss = impl->shared.close,
        .passthrough = config.passthrough,
    });
}

struct ChromeButton::Impl {
    Config config;
    Box background;
    Text label;
    Polyline glyph;
    Menu menu;
    bool hovered = false;
    bool open = false;
    bool menuStale = true;
    bool menuOpen = false;
    Rect menuBounds;

    const MenuEntry* entryAt(const std::vector<U64>& path) const {
        const std::vector<MenuEntry>* level = &config.entries;
        const MenuEntry* entry = nullptr;
        for (const U64 index : path) {
            if (index >= level->size()) {
                return nullptr;
            }
            entry = &(*level)[index];
            level = &entry->children;
        }
        return entry;
    }

    std::vector<MenuEntry> forward(const std::vector<MenuEntry>& entries, const std::vector<U64>& path) const {
        std::vector<MenuEntry> result;
        for (U64 index = 0; index < entries.size(); ++index) {
            auto next = path;
            next.push_back(index);
            MenuEntry entry = entries[index];
            if (entry.onClick) {
                entry.onClick = [this, next]() {
                    if (const auto* current = entryAt(next); current && current->onClick) {
                        current->onClick();
                    }
                };
            }
            entry.children = forward(entries[index].children, next);
            result.push_back(std::move(entry));
        }
        return result;
    }
};

ChromeButton::ChromeButton() {
    impl = std::make_unique<Impl>();
    addChild(impl->background);
    addChild(impl->label);
    addChild(impl->glyph);
    addChild(impl->menu);
}

ChromeButton::~ChromeButton() = default;

void ChromeButton::update(Config config) {
    if (config != impl->config) {
        invalidate(Dirty::Layout);
        impl->menuStale = true;
    }
    impl->config = std::move(config);
    if (impl->config.entries.empty()) {
        impl->open = false;
    }
    impl->label.update({
        .str = impl->config.label,
        .font = impl->config.font,
        .colorKey = impl->config.colorKey,
        .scale = impl->config.scale,
    });
}

bool ChromeButton::menuOpen() const {
    return impl->open;
}

Extent2D<F32> ChromeButton::measure(const Context& ctx, Extent2D<F32> available) {
    const auto& config = impl->config;
    const F32 infinite = std::numeric_limits<F32>::infinity();
    const auto text = measureChild(impl->label, ctx, {infinite, infinite});
    const F32 height = std::round(ResolveLength(ctx, config.height, text.y + 2.0f * Px(ctx, config.padding.y), available.y));
    if (config.glyph != Glyph::None) {
        return FitExtent({height, height}, available);
    }
    return FitExtent({std::ceil(text.x + 2.0f * Px(ctx, config.padding.x)), height}, available);
}

void ChromeButton::layout(const Context& ctx) {
    const Rect bounds = frame();
    const auto& config = impl->config;
    impl->background.update({
        .instances = {{.rect = bounds, .visible = impl->hovered || impl->open,
                       .backgroundColor = ctx.color(config.hoverColorKey)}},
        .clip = bounds,
        .cornerRadius = Px(ctx, config.rounding),
    });
    layoutChild(ctx, impl->background, bounds);

    const F32 infinite = std::numeric_limits<F32>::infinity();
    const auto size = measureChild(impl->label, ctx, {infinite, infinite});
    impl->label.setVisible(config.glyph == Glyph::None);
    layoutChild(ctx, impl->label, {bounds.x + std::floor((bounds.width - size.x) * 0.5f),
                                   bounds.y + std::floor((bounds.height - size.y) * 0.5f), size.x, size.y});

    std::vector<Polyline::Instance> marks;
    if (config.glyph == Glyph::Plus) {
        const F32 arm = std::round(bounds.height * 0.22f);
        const Extent2D<F32> center = {bounds.x + bounds.width * 0.5f, bounds.y + bounds.height * 0.5f};
        const auto color = ctx.color(config.colorKey.empty() ? "text_primary" : config.colorKey);
        const F32 thickness = Px(ctx, 1.5f);
        marks.push_back({.points = {{center.x - arm, center.y}, {center.x + arm, center.y}}, .color = color,
                         .thickness = thickness});
        marks.push_back({.points = {{center.x, center.y - arm}, {center.x, center.y + arm}}, .color = color,
                         .thickness = thickness});
    }
    impl->glyph.update({.instances = std::move(marks), .clip = bounds, .fringe = Fringe(ctx), .capacity = 64});
    layoutChild(ctx, impl->glyph, bounds);

    if (impl->menuStale || impl->menuOpen != impl->open || impl->menuBounds != bounds) {
        impl->menuStale = false;
        impl->menuOpen = impl->open;
        impl->menuBounds = bounds;
        impl->menu.update({
            .entries = impl->forward(config.entries, {}),
            .open = impl->open,
            .placement = PopupPlacement::Below,
            .atCursor = false,
            .passthrough = bounds,
            .onClose = [impl = impl.get(), this]() {
                if (impl->open) {
                    impl->open = false;
                    invalidate(Dirty::Layout);
                }
            },
        });
    }
    layoutChild(ctx, impl->menu, bounds);
}

bool ChromeButton::mouseEvent(const MouseEvent& event) {
    const bool inside = frame().contains(event.position.x, event.position.y);
    switch (event.type) {
        case MouseEventType::Move:
            if (inside != impl->hovered) {
                impl->hovered = inside;
                invalidate(Dirty::Layout);
            }
            mouseEventChildren(event);
            return false;
        case MouseEventType::Click:
            if (event.button == MouseButton::Left && inside) {
                if (impl->config.entries.empty()) {
                    if (impl->config.onClick) {
                        impl->config.onClick();
                    }
                } else {
                    impl->open = !impl->open;
                    invalidate(Dirty::Layout);
                }
                return true;
            }
            return mouseEventChildren(event);
        case MouseEventType::Leave:
        case MouseEventType::Cancel:
            if (impl->hovered) {
                impl->hovered = false;
                invalidate(Dirty::Layout);
            }
            return mouseEventChildren(event);
        default:
            return mouseEventChildren(event);
    }
}

}  // namespace Jetstream::Sakura
