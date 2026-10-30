#include <jetstream/render/sakura/components/checkbox.hh>

#include <jetstream/render/sakura/components/box.hh>
#include <jetstream/render/sakura/components/polyline.hh>
#include <jetstream/render/sakura/components/text.hh>

#include "../context.hh"
#include "../metrics.hh"

#include <algorithm>
#include <cmath>
#include <limits>
#include <optional>
#include <utility>

namespace Jetstream::Sakura {

struct Checkbox::Impl {
    Config config;
    Box square;
    Polyline mark;
    Text label;
    bool hovered = false;
    bool pressed = false;
};

Checkbox::Checkbox() {
    impl = std::make_unique<Impl>();
    addChild(impl->square);
    addChild(impl->mark);
    addChild(impl->label);
}

Checkbox::~Checkbox() = default;

void Checkbox::update(Config config) {
    if (config != impl->config) {
        invalidate(Dirty::Layout);
    }
    impl->label.update({.str = config.label});
    impl->config = std::move(config);
}

Extent2D<F32> Checkbox::measure(const Context& ctx, Extent2D<F32> available) {
    const F32 square = FrameHeightPx(ctx);
    F32 width = square;
    if (!impl->config.label.empty()) {
        const F32 infinite = std::numeric_limits<F32>::infinity();
        width += Px(ctx, kInnerSpacing) + measureChild(impl->label, ctx, {infinite, infinite}).x;
    }
    return FitExtent({width, square}, available);
}

void Checkbox::layout(const Context& ctx) {
    const Rect bounds = frame();
    const F32 square = std::min(bounds.height, FrameHeightPx(ctx));
    const Rect box = {bounds.x, bounds.y + std::floor((bounds.height - square) * 0.5f), square, square};
    const char* key = impl->pressed && impl->hovered ? "frame_bg_active" : impl->hovered ? "frame_bg_hovered" : "card";
    impl->square.update({
        .instances = {{.rect = box, .visible = !bounds.empty(), .backgroundColor = ctx.color(key)}},
        .clip = bounds,
        .cornerRadius = Px(ctx, kFrameRounding),
    });

    const F32 pad = std::max(1.0f, std::floor(square / 6.0f));
    F32 size = square - 2.0f * pad;
    const F32 thickness = std::max(size / 5.0f, 1.0f);
    size -= thickness * 0.5f;
    const F32 originX = box.x + pad + thickness * 0.25f;
    const F32 originY = box.y + pad + thickness * 0.25f;
    const F32 third = size / 3.0f;
    const F32 bx = originX + third;
    const F32 by = originY + size - third * 0.5f;
    impl->mark.update({
        .instances = {{
            .points = {{bx - third, by - third}, {bx, by}, {bx + third * 2.0f, by - third * 2.0f}},
            .color = ctx.color("accent_color"),
            .thickness = thickness,
            .visible = impl->config.value && !bounds.empty(),
        }},
        .clip = bounds,
        .fringe = Fringe(ctx),
        .capacity = 64,
    });

    const F32 labelX = box.right() + Px(ctx, kInnerSpacing);
    const F32 labelHeight = measureChild(impl->label, ctx, {std::numeric_limits<F32>::infinity(), bounds.height}).y;
    layoutChild(ctx, impl->square, bounds);
    layoutChild(ctx, impl->mark, bounds);
    layoutChild(ctx, impl->label, {labelX, box.y + std::floor((square - labelHeight) * 0.5f),
                                   std::max(0.0f, bounds.right() - labelX), labelHeight});
}

bool Checkbox::mouseEvent(const MouseEvent& event) {
    const bool inside = frame().contains(event.position.x, event.position.y);
    switch (event.type) {
        case MouseEventType::Move:
            if (inside != impl->hovered) {
                impl->hovered = inside;
                invalidate(Dirty::Layout);
            }
            return false;
        case MouseEventType::Leave:
        case MouseEventType::Cancel:
            if (impl->hovered || impl->pressed) {
                impl->hovered = false;
                impl->pressed = false;
                invalidate(Dirty::Layout);
            }
            return false;
        case MouseEventType::Click:
            if (!inside || event.button != MouseButton::Left) {
                return false;
            }
            impl->pressed = true;
            invalidate(Dirty::Layout);
            return true;
        case MouseEventType::Release:
            if (event.button != MouseButton::Left || !impl->pressed) {
                return false;
            }
            impl->pressed = false;
            invalidate(Dirty::Layout);
            if (inside && impl->config.onChange) {
                impl->config.onChange(!impl->config.value);
            }
            return true;
        default:
            return false;
    }
}

std::optional<SurfaceCursor> Checkbox::cursorAt(const Extent2D<F32>& point) const {
    if (!frame().contains(point.x, point.y)) {
        return std::nullopt;
    }
    return SurfaceCursor::Hand;
}

}  // namespace Jetstream::Sakura
