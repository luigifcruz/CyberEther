#include <jetstream/render/sakura/components/pill_icon_button.hh>

#include <jetstream/render/sakura/components/box.hh>
#include <jetstream/render/sakura/components/card.hh>
#include <jetstream/render/sakura/components/text.hh>
#include <jetstream/render/sakura/typography.hh>

#include "../context.hh"
#include "../detail/popup_layer.hh"
#include "../detail/helpers.hh"
#include "../metrics.hh"

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>

namespace Jetstream::Sakura {

namespace {

constexpr F32 kIconGlyph = 15.0f;
constexpr F32 kRingAlpha = 0.7f;

}  // namespace

struct PillIconButton::Impl {
    Config config;
    Card card;
    Box halo;
    Text glyph;
    Text tooltip;
    bool hovered = false;
    bool pressed = false;
};

PillIconButton::PillIconButton() {
    impl = std::make_unique<Impl>();
    addChild(impl->card);
    addChild(impl->halo);
    addChild(impl->glyph);
}

PillIconButton::~PillIconButton() = default;

void PillIconButton::update(Config config) {
    if (config != impl->config) {
        invalidate(Dirty::Layout);
    }
    impl->config = std::move(config);
    impl->glyph.update({
        .str = impl->config.icon,
        .colorKey = impl->config.iconColorKey,
        .scale = kIconGlyph / Typography::FontSize,
    });
    impl->tooltip.update({.str = impl->config.tooltip, .scale = Typography::ChromeScale});
    markConfigured();
}

Extent2D<F32> PillIconButton::measure(const Context& ctx, Extent2D<F32> available) {
    const auto& size = impl->config.size;
    const F32 natural = Px(ctx, kPillHeight);
    return {ResolveLength(ctx, size.x, natural, available.x), ResolveLength(ctx, size.y, natural, available.y)};
}

void PillIconButton::layout(const Context& ctx) {
    auto& s = *impl;
    const Rect bounds = frame();
    const F32 infinite = std::numeric_limits<F32>::infinity();
    s.card.update({
        .rects = {bounds},
        .clip = clip(),
        .cornerRadius = std::min(bounds.width, bounds.height) * 0.5f,
        .softness = Px(ctx, kPillShadowSoftness),
        .drop = Px(ctx, kPillShadowDrop),
        .borderWidth = Px(ctx, kPillBorder),
        .lineWidth = Px(ctx, 1.0f),
        .fillKey = kPillFill,
    });
    layoutChild(ctx, s.card, bounds);

    const F32 border = Px(ctx, kPillBorder);
    const bool tinted = s.config.tinted;
    const Rect fill = tinted ? bounds : Inset(bounds, border);
    s.halo.update({
        .instances = {{.rect = fill, .visible = s.hovered && !fill.empty(), .backgroundColor = ctx.color(kPillHover)}},
        .cornerRadius = std::min(fill.width, fill.height) * 0.5f,
        .borderWidth = tinted ? border : 0.0f,
        .borderColor = WithAlpha(ctx.color(s.config.iconColorKey), kRingAlpha),
        .strokeFringe = true,
        .capacity = 1,
    });
    layoutChild(ctx, s.halo, bounds);

    const Extent2D<F32> glyph = measureChild(s.glyph, ctx, {infinite, infinite});
    layoutChild(ctx, s.glyph,
                {bounds.x + std::round((bounds.width - glyph.x) * 0.5f),
                 bounds.y + std::round((bounds.height - glyph.y) * 0.5f), glyph.x, glyph.y},
                bounds);

    if (s.hovered && !s.config.tooltip.empty()) {
        RequestPopup(ctx, {
            .content = &s.tooltip,
            .anchor = bounds,
            .placement = PopupPlacement::BelowEnd,
            .panel = true,
            .padding = kTooltipPadding,
            .interactive = false,
            .delayed = true,
        });
    }
}

bool PillIconButton::mouseEvent(const MouseEvent& event) {
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
            if (event.button != MouseButton::Left) {
                return false;
            }
            s.pressed = inside;
            return inside;
        case MouseEventType::Release: {
            if (event.button != MouseButton::Left) {
                return false;
            }
            const bool pressed = s.pressed;
            s.pressed = false;
            if (pressed && inside && s.config.onClick) {
                s.config.onClick();
            }
            return pressed;
        }
        case MouseEventType::Leave:
        case MouseEventType::Cancel:
            s.pressed = false;
            if (s.hovered) {
                s.hovered = false;
                invalidate(Dirty::Layout);
            }
            return false;
        default:
            return false;
    }
}

std::optional<SurfaceCursor> PillIconButton::cursorAt(const Extent2D<F32>& point) const {
    if (frame().contains(point.x, point.y)) {
        return SurfaceCursor::Hand;
    }
    return std::nullopt;
}

}  // namespace Jetstream::Sakura
