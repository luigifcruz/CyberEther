#include <jetstream/render/sakura/components/menu.hh>

#include <jetstream/render/sakura/components/box.hh>
#include <jetstream/render/sakura/components/card.hh>
#include <jetstream/render/sakura/components/polyline.hh>
#include <jetstream/render/sakura/components/text.hh>
#include <jetstream/render/sakura/components/popup.hh>
#include <jetstream/render/sakura/typography.hh>
#include <jetstream/render/components/text.hh>
#include <jetstream/render/tools/icons_lucide.hh>

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

using Unicode = Jetstream::Render::Components::Text::Unicode;

constexpr F32 kIconGap = 8.0f;
constexpr F32 kMenuInset = 5.0f;
constexpr F32 kRowPadX = 10.0f;
constexpr F32 kRowPadY = 6.0f;
constexpr F32 kMenuRounding = 14.0f;
constexpr F32 kTrailingGap = 24.0f;

std::pair<std::string, std::string> SplitIcon(const std::string& label) {
    U32 codepoint = 0;
    const U64 length = label.empty() ? 0 : Unicode::Decode(label, 0, codepoint);
    if (length == 0 || codepoint < ICON_MIN_LC || codepoint > ICON_MAX_LC) {
        return {std::string(), label};
    }
    U64 rest = length;
    while (rest < label.size() && label[rest] == ' ') {
        ++rest;
    }
    return {label.substr(0, length), label.substr(rest)};
}

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
        Text icon;
        Text label;
        Text shortcut;
    };

    Card panel;
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

    MenuBody() = default;

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
            auto [glyph, text] = SplitIcon(entry.label);
            row.icon.update({
                .str = glyph,
                .colorKey = entry.colorKey.empty() ? "text_primary" : entry.colorKey,
                .opacity = opacity,
            });
            row.label.update({
                .str = text,
                .font = Text::Font::Sans,
                .colorKey = entry.colorKey.empty() ? "text_primary" : entry.colorKey,
                .ellipsis = true,
                .opacity = opacity,
            });
            row.shortcut.update({
                .str = entry.shortcut,
                .align = Text::Align::Right,
                .colorKey = "text_secondary",
                .opacity = opacity,
            });
            if (!glyph.empty()) {
                children.push_back(&row.icon);
            }
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
        F32 icon = 0.0f;
        F32 label = 0.0f;
        F32 shortcut = 0.0f;
        F32 mark = 0.0f;
        F32 trailing = 0.0f;
    };

    static bool marked(const MenuEntry& entry) {
        return !entry.children.empty() || entry.selected;
    }

    Columns columns(const Context& ctx) {
        const F32 infinite = std::numeric_limits<F32>::infinity();
        Columns result;
        for (U64 index = 0; index < entries.size(); ++index) {
            if (entries[index].separator) {
                continue;
            }
            if (!rows[index]->icon.config().str.empty()) {
                result.icon = std::max(result.icon, measureChild(rows[index]->icon, ctx, {infinite, infinite}).x);
            }
            result.label = std::max(result.label, measureChild(rows[index]->label, ctx, {infinite, infinite}).x);
            if (!entries[index].shortcut.empty()) {
                result.shortcut = std::max(result.shortcut,
                                           measureChild(rows[index]->shortcut, ctx, {infinite, infinite}).x);
            }
        }
        result.mark = std::floor(Px(ctx, Typography::FontSize) * 1.20f);
        result.trailing = result.mark;
        for (U64 index = 0; index < entries.size(); ++index) {
            const auto& entry = entries[index];
            if (entry.separator || entry.shortcut.empty()) {
                continue;
            }
            const F32 shortcut = measureChild(rows[index]->shortcut, ctx, {infinite, infinite}).x;
            result.trailing = std::max(result.trailing,
                                       shortcut + (marked(entry) ? Px(ctx, kIconGap) + result.mark : 0.0f));
        }
        if (result.icon > 0.0f) {
            result.icon += Px(ctx, kIconGap);
        }
        return result;
    }

    F32 contentWidth(const Context& ctx, const Columns& cols) const {
        return cols.icon + cols.label + Px(ctx, kTrailingGap) + cols.trailing;
    }

    F32 textHeight(const Context& ctx, U64 index) {
        const F32 infinite = std::numeric_limits<F32>::infinity();
        return measureChild(rows[index]->label, ctx, {infinite, infinite}).y;
    }

    static F32 separatorMargin(const Context& ctx) {
        return std::max(0.0f, Px(ctx, kMenuInset) - Px(ctx, 1.0f));
    }

    F32 rowHeight(const Context& ctx, U64 index) {
        if (entries[index].separator) {
            return Hairline(ctx) + 2.0f * separatorMargin(ctx);
        }
        return textHeight(ctx, index) + 2.0f * Px(ctx, kRowPadY);
    }

    F32 contentHeight(const Context& ctx) {
        F32 height = 0.0f;
        for (U64 index = 0; index < entries.size(); ++index) {
            height += rowHeight(ctx, index);
        }
        return height;
    }

    Extent2D<F32> measure(const Context& ctx, Extent2D<F32> available) override {
        const F32 inset = Px(ctx, kMenuInset);
        const auto cols = columns(ctx);
        return FitExtent({std::ceil(contentWidth(ctx, cols) + 2.0f * (inset + Px(ctx, kRowPadX))),
                          std::ceil(contentHeight(ctx) + 2.0f * inset)}, available);
    }

    void layout(const Context& ctx) override {
        const Rect bounds = frame();
        const F32 font = Px(ctx, Typography::FontSize);
        const F32 line = Hairline(ctx);
        const auto cols = columns(ctx);
        const F32 inset = Px(ctx, kMenuInset);
        const F32 padX = Px(ctx, kRowPadX);
        const F32 padY = Px(ctx, kRowPadY);

        panel.update({
            .rects = bounds.empty() ? std::vector<Rect>{} : std::vector<Rect>{bounds},
            .clip = clip(),
            .cornerRadius = Px(ctx, kMenuRounding),
            .softness = Px(ctx, kFloatingShadowSoftness),
            .drop = Px(ctx, kFloatingShadowDrop),
            .borderWidth = Px(ctx, 1.0f),
            .lineWidth = Px(ctx, 1.0f),
            .fillKey = kPillFill,
            .shadowKey = kFloatingShadow,
            .capacity = 1,
        });
        layoutChild(ctx, panel, bounds);

        std::vector<Box::Instance> highlightInstances;
        std::vector<Box::Instance> separatorInstances;
        std::vector<Polyline::Instance> markInstances;
        rowRects.assign(entries.size(), Rect{});
        F32 y = bounds.y + inset;
        const F32 x = bounds.x + inset + padX;
        const F32 rowWidth = std::max(0.0f, bounds.width - 2.0f * inset);
        const F32 right = bounds.right() - inset - padX;
        childIndex = openSubmenu.has_value() && submenu(*openSubmenu) ? openSubmenu : std::nullopt;
        for (U64 index = 0; index < entries.size(); ++index) {
            const auto& entry = entries[index];
            if (entry.separator) {
                separatorInstances.push_back({
                    .rect = {x, std::round(y + separatorMargin(ctx)), std::max(0.0f, rowWidth - 2.0f * padX), line},
                    .backgroundColor = ctx.color("separator"),
                });
                y += rowHeight(ctx, index);
                continue;
            }
            const Rect highlight = {bounds.x + inset, y, rowWidth, rowHeight(ctx, index)};
            const F32 height = textHeight(ctx, index);
            const F32 top = y + padY;
            rowRects[index] = highlight;
            const bool open = childIndex == index;
            const bool isHovered = hovered == index && entry.enabled;
            if (isHovered || open) {
                const char* key = isHovered && pressed == index ? "header_active" : isHovered ? "header_hovered" : "header";
                highlightInstances.push_back({.rect = highlight, .backgroundColor = ctx.color(key)});
            }
            auto& row = *rows[index];
            if (!row.icon.config().str.empty()) {
                const F32 infinite = std::numeric_limits<F32>::infinity();
                const F32 glyph = measureChild(row.icon, ctx, {infinite, infinite}).x;
                layoutChild(ctx, row.icon, {x, top, glyph, height}, bounds);
            }
            const F32 labelX = x + cols.icon;
            const F32 labelWidth = std::clamp(right - cols.trailing - Px(ctx, kTrailingGap) - labelX, 0.0f, cols.label);
            layoutChild(ctx, row.label, {labelX, top, labelWidth, height}, bounds);
            if (!entry.shortcut.empty()) {
                const F32 shortcutRight = right - (marked(entry) ? cols.mark + Px(ctx, kIconGap) : 0.0f);
                layoutChild(ctx, row.shortcut, {shortcutRight - cols.shortcut, top, cols.shortcut, height}, bounds);
            }
            auto color = ctx.color(entry.colorKey.empty() ? "text_primary" : entry.colorKey);
            color = WithAlpha(color, entry.enabled ? 1.0f : kDisabledAlpha);
            if (!entry.children.empty()) {
                markInstances.push_back({
                    .points = RightArrow({right - font * 0.80f, top}, font),
                    .color = color,
                    .closed = true,
                    .filled = true,
                });
            } else if (entry.selected) {
                F32 thickness = 1.0f;
                auto points = CheckMark({right - font * 0.866f, top + font * 0.134f * 0.5f}, font * 0.866f,
                                        thickness);
                markInstances.push_back({.points = std::move(points), .color = color, .thickness = thickness});
            }
            y += highlight.height;
        }
        highlights.update({
            .instances = std::move(highlightInstances),
            .clip = bounds,
            .cornerRadius = std::max(0.0f, Px(ctx, kMenuRounding) - inset),
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

}  // namespace Jetstream::Sakura
