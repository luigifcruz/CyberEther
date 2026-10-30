#include <jetstream/render/sakura/components/node/surface.hh>

#include <jetstream/render/base/texture.hh>
#include <jetstream/render/sakura/components/box.hh>
#include <jetstream/render/sakura/components/image.hh>
#include <jetstream/render/sakura/components/text.hh>
#include <jetstream/render/sakura/components/surface_image.hh>
#include <jetstream/render/tools/icons_fa.hh>

#include "../../detail/helpers.hh"
#include "../../metrics.hh"
#include "pointer.hh"

#include <cmath>
#include <limits>
#include <memory>
#include <utility>

namespace Jetstream::Sakura {

struct NodeSurface::Impl {
    Config config;
    SurfaceImage surface;
    Box button;
    Text icon;
    Rect buttonRect;
    Pointer pointer;
    bool hovered = false;
    bool buttonHovered = false;
};

NodeSurface::NodeSurface() {
    this->impl = std::make_shared<Impl>();
    addChild(impl->surface);
    addChild(impl->button);
    addChild(impl->icon);
}

NodeSurface::~NodeSurface() = default;

void NodeSurface::update(Config config) {
    if (config != impl->config) {
        invalidate(Dirty::Layout);
    }
    impl->config = std::move(config);
    impl->surface.update({
        .texture = impl->config.texture,
        .rounding = kFrameRounding,
        .onSize = impl->config.onSize,
    });
    impl->icon.update({
        .str = ICON_FA_UP_RIGHT_AND_DOWN_LEFT_FROM_CENTER,
        .colorKey = "action_btn_text",
    });
}

Extent2D<F32> NodeSurface::measure(const Context& ctx, Extent2D<F32> available) {
    return {FillLength(available.x), FitLength(Px(ctx, impl->config.minimumHeight), available.y)};
}

void NodeSurface::layout(const Context& ctx) {
    const Rect bounds = frame();
    const Rect clipRect = Intersect(bounds, clip());
    impl->hovered = impl->pointer.inside(clipRect);

    const F32 buttonSize = Px(ctx, 24.0f);
    const F32 buttonPadding = Px(ctx, 8.0f);
    impl->buttonRect = {bounds.right() - buttonSize - buttonPadding, bounds.y + buttonPadding, buttonSize, buttonSize};
    impl->buttonHovered = impl->config.detachOverlay && impl->hovered && impl->pointer.inside(impl->buttonRect);

    layoutChild(ctx, impl->surface, bounds);

    const bool showButton = impl->config.detachOverlay && impl->hovered;
    impl->button.update({
        .instances = {{
            .rect = impl->buttonRect,
            .visible = showButton,
            .backgroundColor = ctx.color(impl->buttonHovered ? "node_overlay_button_hovered" : "node_overlay_button"),
        }},
        .clip = bounds,
        .cornerRadius = Px(ctx, 4.0f),
    });
    layoutChild(ctx, impl->button, bounds);

    impl->icon.setVisible(showButton);
    const auto iconSize = measureChild(impl->icon, ctx, {std::numeric_limits<F32>::infinity(), std::numeric_limits<F32>::infinity()});
    layoutChild(ctx, impl->icon, {
        impl->buttonRect.x + std::floor((impl->buttonRect.width - iconSize.x) * 0.5f),
        impl->buttonRect.y + std::floor((impl->buttonRect.height - iconSize.y) * 0.5f),
        iconSize.x,
        iconSize.y,
    });
}

bool NodeSurface::mouseEvent(const MouseEvent& event) {
    if (impl->pointer.track(event, Intersect(frame(), clip()), {frame(), impl->buttonRect})) {
        invalidate(Dirty::Layout);
    }
    if (!impl->config.detachOverlay || event.type != MouseEventType::Click || event.button != MouseButton::Left) {
        return false;
    }
    if (!impl->hovered || !impl->buttonRect.contains(event.position.x, event.position.y)) {
        return false;
    }
    if (impl->config.onDetach) {
        impl->config.onDetach();
    }
    return true;
}

}  // namespace Jetstream::Sakura
