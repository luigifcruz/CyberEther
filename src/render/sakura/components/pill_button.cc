#include <jetstream/render/sakura/components/pill_button.hh>

#include <jetstream/render/sakura/components/box.hh>
#include <jetstream/render/sakura/components/text.hh>
#include <jetstream/render/sakura/typography.hh>

#include "../context.hh"
#include "../metrics.hh"

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>
#include <vector>

namespace Jetstream::Sakura {

namespace {

constexpr F32 kPillButtonHeight = 28.0f;
constexpr F32 kPillButtonPadX = 10.0f;
constexpr F32 kPillButtonIconGap = 6.0f;
constexpr F32 kPillButtonIcon = 13.0f;
constexpr F32 kPillButtonText = 14.0f;

}  // namespace

struct PillButton::Impl {
    Config config;
    Box shadow;
    Box fill;
    Box highlight;
    Text icon;
    Text label;
    bool hovered = false;

    F32 alpha() const {
        return config.disabled ? kDisabledAlpha : 1.0f;
    }

    void prepare() {
        icon.update({.str = config.icon, .colorKey = config.colorKey,
                     .scale = kPillButtonIcon / Typography::FontSize, .opacity = alpha()});
        label.update({.str = config.label, .font = Text::Font::Display, .colorKey = config.colorKey,
                      .scale = kPillButtonText / Typography::FontSize, .opacity = alpha()});
    }
};

PillButton::PillButton() {
    impl = std::make_unique<Impl>();
    mountChildren({&impl->shadow, &impl->fill, &impl->highlight, &impl->icon, &impl->label});
}

PillButton::~PillButton() = default;

void PillButton::update(Config config) {
    if (config != impl->config) {
        invalidate(Dirty::Layout);
    }
    impl->config = std::move(config);
    if (impl->config.disabled) {
        impl->hovered = false;
    }
    impl->prepare();
    markConfigured();
}

Extent2D<F32> PillButton::measure(const Context& ctx, Extent2D<F32> available) {
    const F32 infinite = std::numeric_limits<F32>::infinity();
    const F32 height = ResolveLength(ctx, impl->config.height, Px(ctx, kPillButtonHeight), available.y);
    if (impl->config.label.empty()) {
        return FitExtent({height, height}, available);
    }
    F32 width = 2.0f * Px(ctx, kPillButtonPadX) + measureChild(impl->label, ctx, {infinite, infinite}).x;
    if (!impl->config.icon.empty()) {
        width += measureChild(impl->icon, ctx, {infinite, infinite}).x + Px(ctx, kPillButtonIconGap);
    }
    return {FitLength(std::ceil(width), available.x), height};
}

void PillButton::layout(const Context& ctx) {
    auto& s = *impl;
    const Rect bounds = frame();
    const F32 infinite = std::numeric_limits<F32>::infinity();
    s.fill.update({
        .instances = {{.rect = bounds,
                       .backgroundColor = WithAlpha(ctx.color(s.hovered ? s.config.hoverColorKey
                                                                         : s.config.backgroundColorKey),
                                                    s.alpha())}},
        .cornerRadius = bounds.height * 0.5f,
        .capacity = 1,
    });
    layoutChild(ctx, s.fill, bounds);
    const F32 softness = Px(ctx, kPillShadowSoftness);
    const F32 drop = Px(ctx, kPillShadowDrop);
    s.shadow.update({
        .instances = {{.rect = {bounds.x - softness, bounds.y + drop - softness, bounds.width + 2.0f * softness,
                                bounds.height + 2.0f * softness},
                       .visible = s.config.raised,
                       .backgroundColor = WithAlpha(ctx.color("tile_shadow"), s.alpha())}},
        .cornerRadius = bounds.height * 0.5f,
        .softness = softness,
        .capacity = 1,
    });
    layoutChild(ctx, s.shadow, bounds);
    s.highlight.update({
        .instances = {{.rect = bounds, .visible = s.config.raised, .backgroundColor = {0.0f, 0.0f, 0.0f, 0.0f}}},
        .cornerRadius = bounds.height * 0.5f,
        .borderWidth = Px(ctx, 1.0f),
        .borderColor = WithAlpha(ctx.color("action_btn_highlight"), s.alpha()),
        .strokeFringe = true,
        .topLit = true,
        .capacity = 1,
    });
    layoutChild(ctx, s.highlight, bounds);
    const bool iconOnly = s.config.label.empty();
    F32 x = bounds.x + Px(ctx, kPillButtonPadX);
    const auto place = [&](Text& text) {
        const Extent2D<F32> size = measureChild(text, ctx, {infinite, infinite});
        if (iconOnly) {
            x = std::round(bounds.x + (bounds.width - size.x) * 0.5f);
        }
        layoutChild(ctx, text, {x, std::round(bounds.y + (bounds.height - size.y) * 0.5f), size.x, size.y}, bounds);
        x += size.x;
    };
    if (!s.config.icon.empty()) {
        place(s.icon);
        x += Px(ctx, kPillButtonIconGap);
    }
    s.icon.setVisible(!s.config.icon.empty());
    s.label.setVisible(!iconOnly);
    if (!iconOnly) {
        place(s.label);
    }
}

bool PillButton::mouseEvent(const MouseEvent& event) {
    auto& s = *impl;
    if (s.config.disabled) {
        return false;
    }
    const bool inside = frame().contains(event.position.x, event.position.y);
    switch (event.type) {
        case MouseEventType::Move:
            if (inside != s.hovered) {
                s.hovered = inside;
                invalidate(Dirty::Layout);
            }
            return false;
        case MouseEventType::Click:
            if (inside && event.button == MouseButton::Left && s.config.onClick) {
                s.config.onClick();
            }
            return inside;
        case MouseEventType::Leave:
        case MouseEventType::Cancel:
            if (s.hovered) {
                s.hovered = false;
                invalidate(Dirty::Layout);
            }
            return false;
        default:
            return inside;
    }
}

std::optional<SurfaceCursor> PillButton::cursorAt(const Extent2D<F32>& point) const {
    if (!impl->config.disabled && frame().contains(point.x, point.y)) {
        return SurfaceCursor::Hand;
    }
    return std::nullopt;
}

}  // namespace Jetstream::Sakura
