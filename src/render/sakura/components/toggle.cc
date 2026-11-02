#include <jetstream/render/sakura/components/toggle.hh>

#include <jetstream/render/sakura/components/box.hh>

#include "../context.hh"
#include "../metrics.hh"

#include <cmath>
#include <utility>

namespace Jetstream::Sakura {

namespace {

constexpr F32 kToggleWidth = 38.0f;
constexpr F32 kToggleHeight = 22.0f;
constexpr F32 kToggleInset = 2.0f;

}  // namespace

struct Toggle::Impl {
    Config config;
    Box track;
    Box knob;
    bool hovered = false;
    bool pressed = false;
};

Toggle::Toggle() {
    impl = std::make_unique<Impl>();
    mountChildren({&impl->track, &impl->knob});
}

Toggle::~Toggle() = default;

void Toggle::update(Config config) {
    if (config != impl->config) {
        invalidate(Dirty::Layout);
    }
    impl->config = std::move(config);
    if (impl->config.disabled) {
        impl->hovered = false;
        impl->pressed = false;
    }
}

Extent2D<F32> Toggle::measure(const Context& ctx, Extent2D<F32> available) {
    return FitExtent({Px(ctx, kToggleWidth), Px(ctx, kToggleHeight)}, available);
}

void Toggle::layout(const Context& ctx) {
    const auto& config = impl->config;
    const Rect bounds = frame();
    const F32 alpha = config.disabled ? kDisabledAlpha : 1.0f;
    const F32 height = std::min(bounds.height, Px(ctx, kToggleHeight));
    const Rect track = {bounds.x, bounds.y + std::floor((bounds.height - height) * 0.5f), bounds.width, height};
    const F32 inset = Px(ctx, kToggleInset);
    const F32 diameter = std::max(0.0f, std::min(height, track.width) - 2.0f * inset);
    const F32 knobX = config.value ? std::max(track.x + inset, track.right() - inset - diameter) : track.x + inset;
    const char* trackKey = config.value ? (impl->hovered ? "toggle_on_hovered" : "toggle_on")
                                        : (impl->hovered ? "toggle_off_hovered" : "toggle_off");

    impl->track.update({
        .instances = {{.rect = track, .visible = !track.empty(), .backgroundColor = WithAlpha(ctx.color(trackKey), alpha)}},
        .cornerRadius = height * 0.5f,
        .capacity = 1,
    });
    layoutChild(ctx, impl->track, bounds);
    impl->knob.update({
        .instances = {{.rect = {knobX, track.y + inset, diameter, diameter},
                       .visible = diameter > 0.0f,
                       .backgroundColor = WithAlpha(ctx.color("toggle_knob"), alpha)}},
        .cornerRadius = diameter * 0.5f,
        .capacity = 1,
    });
    layoutChild(ctx, impl->knob, bounds);
}

bool Toggle::mouseEvent(const MouseEvent& event) {
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
            if (inside && s.config.onChange) {
                s.config.onChange(!s.config.value);
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

std::optional<SurfaceCursor> Toggle::cursorAt(const Extent2D<F32>& point) const {
    if (!impl->config.disabled && frame().contains(point.x, point.y)) {
        return SurfaceCursor::Hand;
    }
    return std::nullopt;
}

}  // namespace Jetstream::Sakura
