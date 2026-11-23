#include <jetstream/render/sakura/components/badge.hh>

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

constexpr F32 kBadgeHeight = 34.0f;
constexpr F32 kBadgePadX = 14.0f;
constexpr F32 kBadgeText = 14.0f;
constexpr F32 kIconSize = 13.0f;

}  // namespace

struct Badge::Impl {
    Config config;
    Box pill;
    Text label;
    Text icon;
    Rect iconRect;
    bool hovered = false;
    bool pressed = false;

    bool hasIcon() const {
        return !config.icon.empty();
    }

    void updateIcon() {
        icon.update({
            .str = config.icon,
            .colorKey = hovered ? config.iconHoveredColorKey : config.textColorKey,
            .scale = kIconSize / Typography::FontSize,
        });
    }
};

Badge::Badge() {
    impl = std::make_unique<Impl>();
    mountChildren({&impl->pill, &impl->label, &impl->icon});
}

Badge::~Badge() = default;

void Badge::update(Config config) {
    if (config != impl->config) {
        invalidate(Dirty::Layout);
    }
    impl->config = std::move(config);
    impl->label.update({
        .str = impl->config.label,
        .font = Text::Font::Display,
        .colorKey = impl->config.textColorKey,
        .scale = kBadgeText / Typography::FontSize,
        .weight = kLabelWeight,
    });
    if (!impl->hasIcon()) {
        impl->hovered = false;
        impl->pressed = false;
    }
    impl->updateIcon();
}

Extent2D<F32> Badge::measure(const Context& ctx, Extent2D<F32> available) {
    const F32 infinite = std::numeric_limits<F32>::infinity();
    const F32 text = measureChild(impl->label, ctx, {infinite, infinite}).x;
    const F32 icon = impl->hasIcon() ? Px(ctx, kPillIconGap) + measureChild(impl->icon, ctx, {infinite, infinite}).x : 0.0f;
    return FitExtent({std::ceil(text + icon + 2.0f * Px(ctx, kBadgePadX)), Px(ctx, kBadgeHeight)}, available);
}

void Badge::layout(const Context& ctx) {
    auto& s = *impl;
    const Rect bounds = frame();
    const F32 height = std::min(bounds.height, Px(ctx, kBadgeHeight));
    const Rect pill = {bounds.x, bounds.y + std::floor((bounds.height - height) * 0.5f), bounds.width, height};
    s.pill.update({
        .instances = {{.rect = pill, .visible = !pill.empty(), .backgroundColor = ctx.color(s.config.backgroundColorKey)}},
        .cornerRadius = height * 0.5f,
        .borderWidth = Px(ctx, 1.0f),
        .borderColor = ctx.color(s.config.borderColorKey),
        .strokeFringe = true,
        .capacity = 1,
    });
    layoutChild(ctx, s.pill, bounds);

    const F32 infinite = std::numeric_limits<F32>::infinity();
    const Extent2D<F32> text = measureChild(s.label, ctx, {infinite, infinite});
    const Extent2D<F32> icon = s.hasIcon() ? measureChild(s.icon, ctx, {infinite, infinite}) : Extent2D<F32>{};
    const F32 gap = s.hasIcon() ? Px(ctx, kPillIconGap) : 0.0f;
    const F32 pad = Px(ctx, kBadgePadX);
    const F32 content = text.x + gap + icon.x;
    const F32 left = std::round(pill.x + std::max(pad, (pill.width - content) * 0.5f));
    const Rect inner = {pill.x + pad, pill.y, std::max(0.0f, pill.width - 2.0f * pad), pill.height};
    layoutChild(ctx, s.label, {left, std::round(pill.y + (pill.height - text.y) * 0.5f), text.x, text.y},
                {inner.x, inner.y, std::max(0.0f, inner.width - gap - icon.x), inner.height});
    const F32 iconX = std::min(left + text.x + gap, inner.right() - icon.x);
    s.iconRect = s.hasIcon() ? Rect{iconX - gap * 0.5f, pill.y, icon.x + gap, pill.height} : Rect{};
    layoutChild(ctx, s.icon, {iconX, std::round(pill.y + (pill.height - icon.y) * 0.5f), icon.x, icon.y}, inner);
}

bool Badge::mouseEvent(const MouseEvent& event) {
    auto& s = *impl;
    if (!s.hasIcon()) {
        return false;
    }
    const bool inside = frame().contains(event.position.x, event.position.y) &&
                        s.iconRect.contains(event.position.x, event.position.y);
    switch (event.type) {
        case MouseEventType::Move:
            if (inside != s.hovered) {
                s.hovered = inside;
                s.updateIcon();
                invalidate(Dirty::Layout);
            }
            return false;
        case MouseEventType::Leave:
        case MouseEventType::Cancel:
            if (s.hovered || s.pressed) {
                s.hovered = false;
                s.pressed = false;
                s.updateIcon();
                invalidate(Dirty::Layout);
            }
            return false;
        case MouseEventType::Click:
            if (event.button != MouseButton::Left || !inside) {
                return false;
            }
            s.pressed = true;
            invalidate(Dirty::Layout);
            return true;
        case MouseEventType::Release: {
            if (event.button != MouseButton::Left) {
                return false;
            }
            if (!s.pressed) {
                return false;
            }
            s.pressed = false;
            invalidate(Dirty::Layout);
            if (inside && s.config.onIcon) {
                s.config.onIcon();
            }
            return true;
        }
        default:
            return false;
    }
}

std::optional<SurfaceCursor> Badge::cursorAt(const Extent2D<F32>& point) const {
    if (!impl->hasIcon() || !frame().contains(point.x, point.y) || !impl->iconRect.contains(point.x, point.y)) {
        return std::nullopt;
    }
    return SurfaceCursor::Hand;
}

}  // namespace Jetstream::Sakura
