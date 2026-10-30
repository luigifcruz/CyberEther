#include <jetstream/render/sakura/components/hud_layer.hh>

#include <jetstream/render/sakura/components/box.hh>

#include "../context.hh"
#include "../metrics.hh"
#include "../detail/anchor.hh"

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>

namespace Jetstream::Sakura {

struct HudLayer::Impl {
    Config config;
    std::vector<std::unique_ptr<Box>> frames;
    std::vector<Rect> rects;
};

HudLayer::HudLayer() {
    impl = std::make_unique<Impl>();
}

HudLayer::~HudLayer() = default;

void HudLayer::update(Config config) {
    if (config != impl->config) {
        invalidate(Dirty::Layout);
    }
    impl->config = std::move(config);
    while (impl->frames.size() < impl->config.huds.size()) {
        impl->frames.push_back(std::make_unique<Box>());
    }
    std::vector<Component*> children;
    for (U64 index = 0; index < impl->config.huds.size(); ++index) {
        children.push_back(impl->frames[index].get());
        if (impl->config.huds[index].content) {
            children.push_back(impl->config.huds[index].content);
        }
    }
    mountChildren(children);
}

bool HudLayer::covers(const Extent2D<F32>& point) const {
    for (const auto& rect : impl->rects) {
        if (rect.contains(point.x, point.y)) {
            return true;
        }
    }
    return false;
}

void HudLayer::layout(const Context& ctx) {
    const Rect bounds = frame();
    const F32 infinite = std::numeric_limits<F32>::infinity();
    impl->rects.clear();
    for (U64 index = 0; index < impl->config.huds.size(); ++index) {
        const auto& hud = impl->config.huds[index];
        const Extent2D<F32> padding = {Px(ctx, hud.padding.x), Px(ctx, hud.padding.y)};
        const F32 margin = Px(ctx, hud.margin);
        const Extent2D<F32> room = {std::max(0.0f, bounds.width - 2.0f * margin),
                                    std::max(0.0f, bounds.height - 2.0f * margin)};
        Extent2D<F32> size = {ResolveLength(ctx, hud.size.x, 0.0f, room.x), ResolveLength(ctx, hud.size.y, 0.0f, room.y)};
        if (hud.content && (hud.size.x == 0.0f || hud.size.y == 0.0f)) {
            const F32 width = hud.size.x != 0.0f ? size.x - 2.0f * padding.x : std::max(0.0f, room.x - 2.0f * padding.x);
            const auto measured = measureChild(*hud.content, ctx, {width, infinite});
            if (hud.size.x == 0.0f) {
                size.x = std::ceil(measured.x + 2.0f * padding.x);
            }
            if (hud.size.y == 0.0f) {
                size.y = std::ceil(measured.y + 2.0f * padding.y);
            }
        }
        const Rect rect = Anchored(bounds, hud.anchor, size, {Px(ctx, hud.offset.x), Px(ctx, hud.offset.y)}, margin);
        impl->rects.push_back(rect);
        impl->frames[index]->update({
            .instances = {{.rect = rect, .visible = !rect.empty(), .backgroundColor = ctx.color(hud.backgroundColorKey)}},
            .cornerRadius = Px(ctx, hud.rounding),
            .borderWidth = hud.borderSize > 0.0f ? Px(ctx, hud.borderSize * 2.0f) : 0.0f,
            .borderColor = ctx.color(hud.borderColorKey),
            .strokeFringe = true,
        });
        layoutChild(ctx, *impl->frames[index], rect);
        if (hud.content) {
            layoutChild(ctx, *hud.content, {rect.x + padding.x, rect.y + padding.y,
                                            std::max(0.0f, rect.width - 2.0f * padding.x),
                                            std::max(0.0f, rect.height - 2.0f * padding.y)});
        }
    }
}

bool HudLayer::mouseEvent(const MouseEvent& event) {
    const bool handled = mouseEventChildren(event);
    if (handled || event.type != MouseEventType::Click || event.button != MouseButton::Left) {
        return handled;
    }
    for (U64 index = 0; index < impl->rects.size() && index < impl->config.huds.size(); ++index) {
        const auto& hud = impl->config.huds[index];
        if (hud.clickable && hud.onClick && impl->rects[index].contains(event.position.x, event.position.y)) {
            hud.onClick();
            return true;
        }
    }
    return false;
}

std::optional<SurfaceCursor> HudLayer::cursorAt(const Extent2D<F32>& point) const {
    for (U64 index = 0; index < impl->rects.size() && index < impl->config.huds.size(); ++index) {
        if (impl->config.huds[index].clickable && impl->rects[index].contains(point.x, point.y)) {
            return SurfaceCursor::Hand;
        }
    }
    return std::nullopt;
}

}  // namespace Jetstream::Sakura
