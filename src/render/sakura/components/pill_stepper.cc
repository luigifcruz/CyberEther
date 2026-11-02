#include <jetstream/render/sakura/components/pill_stepper.hh>

#include <jetstream/render/sakura/components/box.hh>
#include <jetstream/render/sakura/components/card.hh>
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

constexpr F32 kStepperMark = 12.0f;
constexpr F32 kStepperText = 14.0f;
constexpr F32 kStepperStroke = 1.5f;
constexpr F32 kStepperButton = 28.0f;
constexpr F32 kStepperInset = 3.0f;
constexpr F32 kStepperHalo = 24.0f;

}  // namespace

struct PillStepper::Impl {
    enum class Part {
        None,
        Decrement,
        Value,
        Increment,
    };

    Config config;
    Card card;
    Box hover;
    Box marks;
    Text value;
    Rect decrementRect;
    Rect valueRect;
    Rect incrementRect;
    Part hovered = Part::None;
    Part pressed = Part::None;

    Part at(const Extent2D<F32>& point) const {
        if (config.canDecrement && decrementRect.contains(point.x, point.y)) {
            return Part::Decrement;
        }
        if (config.canIncrement && incrementRect.contains(point.x, point.y)) {
            return Part::Increment;
        }
        if (config.onReset && valueRect.contains(point.x, point.y)) {
            return Part::Value;
        }
        return Part::None;
    }

    static const char* colorFor(bool enabled) {
        return enabled ? "text_secondary" : "text_disabled";
    }
};

PillStepper::PillStepper() {
    impl = std::make_unique<Impl>();
    std::vector<Component*> children = {&impl->card};
    children.push_back(&impl->hover);
    children.push_back(&impl->marks);
    children.push_back(&impl->value);
    mountChildren(children);
}

PillStepper::~PillStepper() = default;

void PillStepper::update(Config config) {
    if (config != impl->config) {
        invalidate(Dirty::Layout);
    }
    impl->config = std::move(config);
    impl->value.update({.str = impl->config.label, .font = Text::Font::Display, .colorKey = "text_primary",
                        .scale = kStepperText / Typography::FontSize});
    markConfigured();
}

Extent2D<F32> PillStepper::measure(const Context& ctx, Extent2D<F32> available) {
    const auto& size = impl->config.size;
    const F32 height = ResolveLength(ctx, size.y, Px(ctx, kPillHeight), available.y);
    if (size.x > 0.0f) {
        return {ResolveLength(ctx, size.x, 0.0f, available.x), height};
    }
    const F32 infinite = std::numeric_limits<F32>::infinity();
    const F32 label = measureChild(impl->value, ctx, {infinite, infinite}).x;
    const F32 width = std::ceil(2.0f * (Px(ctx, kStepperButton) + Px(ctx, kStepperInset)) + label + Px(ctx, kPillIconGap));
    return {ResolveLength(ctx, size.x, width, available.x), height};
}

void PillStepper::layout(const Context& ctx) {
    auto& s = *impl;
    const Rect bounds = frame();
    const F32 infinite = std::numeric_limits<F32>::infinity();
    s.card.update({
        .rects = {bounds},
        .clip = clip(),
        .cornerRadius = bounds.height * 0.5f,
        .softness = Px(ctx, kPillShadowSoftness),
        .drop = Px(ctx, kPillShadowDrop),
        .borderWidth = Px(ctx, kPillBorder),
        .lineWidth = Px(ctx, 1.0f),
        .fillKey = kPillFill,
    });
    layoutChild(ctx, s.card, bounds);
    const F32 inset = Px(ctx, kStepperInset);
    const F32 button = Px(ctx, kStepperButton);
    s.decrementRect = {bounds.x + inset, bounds.y, button, bounds.height};
    s.incrementRect = {bounds.right() - inset - button, bounds.y, button, bounds.height};
    s.valueRect = {s.decrementRect.right(), bounds.y, std::max(0.0f, s.incrementRect.x - s.decrementRect.right()),
                   bounds.height};

    const F32 length = std::round(Px(ctx, kStepperMark));
    const F32 thickness = std::max(1.0f, std::round(Px(ctx, kStepperStroke)));
    const auto bar = [&](const Rect& rect, bool vertical) {
        const F32 cx = std::round(rect.x + rect.width * 0.5f);
        const F32 cy = std::round(rect.y + rect.height * 0.5f);
        return vertical ? Rect{cx - std::floor(thickness * 0.5f), cy - std::floor(length * 0.5f), thickness, length}
                        : Rect{cx - std::floor(length * 0.5f), cy - std::floor(thickness * 0.5f), length, thickness};
    };
    const auto down = ctx.color(Impl::colorFor(s.config.canDecrement));
    const auto up = ctx.color(Impl::colorFor(s.config.canIncrement));
    const F32 halo = Px(ctx, kStepperHalo);
    std::vector<Box::Instance> hoverInstances;
    const auto circle = [&](const Rect& rect) {
        return Rect{std::round(rect.x + (rect.width - halo) * 0.5f), std::round(rect.y + (rect.height - halo) * 0.5f),
                    halo, halo};
    };
    if (s.hovered == Impl::Part::Decrement) {
        hoverInstances.push_back({.rect = circle(s.decrementRect), .backgroundColor = ctx.color(kPillHover)});
    } else if (s.hovered == Impl::Part::Increment) {
        hoverInstances.push_back({.rect = circle(s.incrementRect), .backgroundColor = ctx.color(kPillHover)});
    } else if (s.hovered == Impl::Part::Value) {
        hoverInstances.push_back({
            .rect = {s.valueRect.x, std::round(s.valueRect.y + (s.valueRect.height - halo) * 0.5f), s.valueRect.width,
                     halo},
            .backgroundColor = ctx.color(kPillHover),
        });
    }
    s.hover.update({.instances = std::move(hoverInstances), .cornerRadius = halo * 0.5f,
                    .capacity = 1});
    layoutChild(ctx, s.hover, bounds);
    s.marks.update({
        .instances = {
            {.rect = bar(s.decrementRect, false), .backgroundColor = down},
            {.rect = bar(s.incrementRect, false), .backgroundColor = up},
            {.rect = bar(s.incrementRect, true), .backgroundColor = up},
        },
        .cornerRadius = thickness * 0.5f,
        .capacity = 3,
    });
    layoutChild(ctx, s.marks, bounds);
    const auto center = [&](Text& text, const Rect& rect) {
        const Extent2D<F32> size = measureChild(text, ctx, {infinite, infinite});
        layoutChild(ctx, text, {rect.x + (rect.width - size.x) * 0.5f, rect.y + (rect.height - size.y) * 0.5f, size.x,
                                size.y},
                    rect);
    };
    center(s.value, s.valueRect);
}

bool PillStepper::mouseEvent(const MouseEvent& event) {
    auto& s = *impl;
    const bool inside = frame().contains(event.position.x, event.position.y);
    switch (event.type) {
        case MouseEventType::Move: {
            const auto part = s.at(event.position);
            if (part != s.hovered) {
                s.hovered = part;
                invalidate(Dirty::Layout);
            }
            return false;
        }
        case MouseEventType::Click:
            if (event.button == MouseButton::Left) {
                s.pressed = s.at(event.position);
            }
            return inside;
        case MouseEventType::Release: {
            if (event.button != MouseButton::Left) {
                return false;
            }
            const auto pressed = s.pressed;
            s.pressed = Impl::Part::None;
            if (pressed != Impl::Part::None && pressed == s.at(event.position)) {
                const auto& action = pressed == Impl::Part::Decrement ? s.config.onDecrement
                                     : pressed == Impl::Part::Increment ? s.config.onIncrement
                                                                       : s.config.onReset;
                if (action) {
                    action();
                }
            }
            return pressed != Impl::Part::None;
        }
        case MouseEventType::Leave:
        case MouseEventType::Cancel:
            s.pressed = Impl::Part::None;
            if (s.hovered != Impl::Part::None) {
                s.hovered = Impl::Part::None;
                invalidate(Dirty::Layout);
            }
            return false;
        default:
            return inside;
    }
}

std::optional<SurfaceCursor> PillStepper::cursorAt(const Extent2D<F32>& point) const {
    if (impl->at(point) != Impl::Part::None) {
        return SurfaceCursor::Hand;
    }
    return std::nullopt;
}

}  // namespace Jetstream::Sakura
