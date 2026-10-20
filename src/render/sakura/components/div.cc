#include <jetstream/render/sakura/components/div.hh>

#include <jetstream/render/sakura/components/box.hh>
#include <jetstream/render/sakura/components/card.hh>

#include "../context.hh"
#include "../metrics.hh"

#include <algorithm>
#include <cmath>
#include <limits>
#include <optional>
#include <utility>
#include <vector>

namespace Jetstream::Sakura {

struct Div::Impl {
    Config config;
    Box frame;
    Card card;
    bool hovered = false;

    bool interactive() const {
        return static_cast<bool>(config.onClick) || static_cast<bool>(config.onDoubleClick);
    }
};

Div::Div() {
    impl = std::make_unique<Impl>();
    addChild(impl->frame);
    addChild(impl->card);
}

Div::~Div() = default;

void Div::update(Config config) {
    if (config != impl->config) {
        invalidate(Dirty::Layout);
    }
    std::vector<Component*> children = {&impl->frame, &impl->card};
    if (config.content) {
        children.push_back(config.content);
    }
    setChildren(children);
    impl->config = std::move(config);
}

Extent2D<F32> Div::measure(const Context& ctx, Extent2D<F32> available) {
    const auto& config = impl->config;
    const F32 padding = Px(ctx, config.padding);
    const F32 infinite = std::numeric_limits<F32>::infinity();
    const bool natural = config.size.x == 0.0f || (config.size.x < 0.0f && !std::isfinite(available.x));
    F32 width = natural ? infinite : ResolveLength(ctx, config.size.x, 0.0f, available.x);
    Extent2D<F32> content = {0.0f, 0.0f};
    if (config.content) {
        const F32 inner = std::isfinite(width) ? std::max(0.0f, width - 2.0f * padding) : infinite;
        content = measureChild(*config.content, ctx, {inner, infinite});
    }
    if (!std::isfinite(width)) {
        width = FitLength(content.x + 2.0f * padding, available.x);
        if (config.content && width < content.x + 2.0f * padding) {
            content = measureChild(*config.content, ctx, {std::max(0.0f, width - 2.0f * padding), infinite});
        }
    }
    return {width, ResolveLength(ctx, config.size.y, content.y + 2.0f * padding, available.y)};
}

void Div::layout(const Context& ctx) {
    const auto& config = impl->config;
    const Rect bounds = frame();
    const F32 padding = Px(ctx, config.padding);
    const bool highlighted = config.selected || (impl->hovered && impl->interactive());
    const auto key = config.selected ? config.selectedColorKey : highlighted ? config.hoveredColorKey : config.colorKey;
    impl->frame.setVisible(!config.chrome);
    impl->frame.update({
        .instances = {{.rect = bounds, .visible = !bounds.empty(), .backgroundColor = ctx.color(key)}},
        .clip = bounds,
        .cornerRadius = Px(ctx, config.rounding),
        .borderWidth = config.border ? Px(ctx, config.borderWidth) : 0.0f,
        .borderColor = ctx.color(config.borderColorKey),
        .strokeFringe = true,
    });
    layoutChild(ctx, impl->frame, bounds);
    impl->card.setVisible(config.chrome);
    impl->card.update({
        .rects = {bounds},
        .clip = clip(),
        .cornerRadius = Px(ctx, config.rounding),
        .borderWidth = config.border ? Px(ctx, config.borderWidth) : 0.0f,
        .lineWidth = Px(ctx, 1.0f),
        .fillKey = key,
        .capacity = 1,
    });
    layoutChild(ctx, impl->card, bounds);
    if (config.content) {
        layoutChild(ctx, *config.content, {bounds.x + padding, bounds.y + padding,
                                           std::max(0.0f, bounds.width - 2.0f * padding),
                                           std::max(0.0f, bounds.height - 2.0f * padding)});
    }
}

bool Div::mouseEvent(const MouseEvent& event) {
    if (mouseEventChildren(event)) {
        return true;
    }
    const bool inside = frame().contains(event.position.x, event.position.y);
    switch (event.type) {
        case MouseEventType::Move:
            if (inside != impl->hovered) {
                impl->hovered = inside;
                if (impl->interactive()) {
                    invalidate(Dirty::Layout);
                }
            }
            return false;
        case MouseEventType::Leave:
        case MouseEventType::Cancel:
            if (impl->hovered) {
                impl->hovered = false;
                if (impl->interactive()) {
                    invalidate(Dirty::Layout);
                }
            }
            return false;
        case MouseEventType::Click:
            if (!inside || event.button != MouseButton::Left || !impl->interactive()) {
                return false;
            }
            if (event.clicks >= 2 && impl->config.onDoubleClick) {
                impl->config.onDoubleClick();
            } else if (impl->config.onClick) {
                impl->config.onClick();
            }
            return true;
        default:
            return false;
    }
}

std::optional<SurfaceCursor> Div::cursorAt(const Extent2D<F32>& point) const {
    if (!frame().contains(point.x, point.y)) {
        return std::nullopt;
    }
    return impl->interactive() ? SurfaceCursor::Hand : SurfaceCursor::Default;
}

}  // namespace Jetstream::Sakura
