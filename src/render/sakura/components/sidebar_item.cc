#include <jetstream/render/sakura/components/sidebar_item.hh>

#include <jetstream/render/sakura/components/box.hh>
#include <jetstream/render/sakura/components/text.hh>
#include <jetstream/render/sakura/typography.hh>

#include "../context.hh"
#include "../metrics.hh"

#include <cmath>
#include <limits>
#include <utility>

namespace Jetstream::Sakura {

namespace {

constexpr F32 kSidebarItemHeight = 30.0f;
constexpr F32 kSidebarItemPadX = 9.0f;
constexpr F32 kSidebarItemIconSlot = 16.0f;
constexpr F32 kSidebarItemGap = 8.0f;
constexpr F32 kSidebarItemRounding = 8.0f;
constexpr F32 kSidebarItemIcon = 13.0f;
constexpr F32 kSidebarItemText = 16.0f;

}  // namespace

struct SidebarItem::Impl {
    Config config;
    Box fill;
    Text icon;
    Text label;
    bool hovered = false;
    bool pressed = false;
};

SidebarItem::SidebarItem() {
    impl = std::make_unique<Impl>();
    mountChildren({&impl->fill, &impl->icon, &impl->label});
}

SidebarItem::~SidebarItem() = default;

void SidebarItem::update(Config config) {
    if (config != impl->config) {
        invalidate(Dirty::Layout);
    }
    impl->config = std::move(config);
    impl->icon.update({
        .str = impl->config.icon,
        .colorKey = "text_primary",
        .scale = kSidebarItemIcon / Typography::FontSize,
    });
    impl->label.update({
        .str = impl->config.label,
        .font = Text::Font::Sans,
        .colorKey = "text_primary",
        .scale = kSidebarItemText / Typography::FontSize,
        .weight = kLabelWeight,
    });
}

Extent2D<F32> SidebarItem::measure(const Context& ctx, Extent2D<F32> available) {
    const F32 infinite = std::numeric_limits<F32>::infinity();
    const F32 label = measureChild(impl->label, ctx, {infinite, infinite}).x;
    const F32 slot = impl->config.icon.empty() ? 0.0f : Px(ctx, kSidebarItemIconSlot) + Px(ctx, kSidebarItemGap);
    const F32 natural = std::ceil(2.0f * Px(ctx, kSidebarItemPadX) + Px(ctx, impl->config.indent) + slot + label);
    return {FillLength(available.x, natural), FitLength(Px(ctx, kSidebarItemHeight), available.y)};
}

void SidebarItem::layout(const Context& ctx) {
    auto& s = *impl;
    const Rect bounds = frame();
    const F32 infinite = std::numeric_limits<F32>::infinity();
    const bool lit = s.config.selected || s.hovered;
    s.fill.update({
        .instances = {{.rect = bounds, .visible = lit && !bounds.empty(),
                       .backgroundColor = ctx.color(s.config.selected ? "header_active" : "header_hovered")}},
        .cornerRadius = Px(ctx, kSidebarItemRounding),
        .capacity = 1,
    });
    layoutChild(ctx, s.fill, bounds);

    const F32 slotX = bounds.x + Px(ctx, kSidebarItemPadX) + Px(ctx, s.config.indent);
    const bool iconed = !s.config.icon.empty();
    const F32 slot = iconed ? Px(ctx, kSidebarItemIconSlot) : 0.0f;
    s.icon.setVisible(iconed);
    if (iconed) {
        const Extent2D<F32> icon = measureChild(s.icon, ctx, {infinite, infinite});
        layoutChild(ctx, s.icon, {std::round(slotX + (slot - icon.x) * 0.5f),
                                  std::round(bounds.y + (bounds.height - icon.y) * 0.5f), icon.x, icon.y}, bounds);
    }

    const F32 labelX = slotX + (iconed ? slot + Px(ctx, kSidebarItemGap) : 0.0f);
    const F32 width = std::max(0.0f, bounds.right() - Px(ctx, kSidebarItemPadX) - labelX);
    const Extent2D<F32> label = measureChild(s.label, ctx, {infinite, infinite});
    layoutChild(ctx, s.label, {labelX, std::round(bounds.y + (bounds.height - label.y) * 0.5f),
                               std::min(width, label.x), label.y}, bounds);
}

bool SidebarItem::mouseEvent(const MouseEvent& event) {
    auto& s = *impl;
    const bool inside = frame().contains(event.position.x, event.position.y);
    switch (event.type) {
        case MouseEventType::Move:
            if (inside != s.hovered) {
                s.hovered = inside;
                invalidate(Dirty::Layout);
            }
            return false;
        case MouseEventType::Click:
            if (!inside || event.button != MouseButton::Left) {
                return false;
            }
            s.pressed = true;
            return true;
        case MouseEventType::Release:
            if (event.button != MouseButton::Left || !s.pressed) {
                return false;
            }
            s.pressed = false;
            if (inside && s.config.onSelect) {
                s.config.onSelect();
            }
            return true;
        case MouseEventType::Leave:
        case MouseEventType::Cancel:
            if (s.hovered || s.pressed) {
                s.hovered = false;
                s.pressed = false;
                invalidate(Dirty::Layout);
            }
            return false;
        default:
            return false;
    }
}

std::optional<SurfaceCursor> SidebarItem::cursorAt(const Extent2D<F32>& point) const {
    if (frame().contains(point.x, point.y)) {
        return SurfaceCursor::Hand;
    }
    return std::nullopt;
}

}  // namespace Jetstream::Sakura
