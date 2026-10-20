#include <jetstream/render/sakura/components/modal_layer.hh>

#include <jetstream/render/sakura/components/box.hh>
#include <jetstream/render/sakura/components/button.hh>
#include <jetstream/render/sakura/components/divider.hh>
#include <jetstream/render/sakura/components/stack.hh>

#include "../context.hh"
#include "../metrics.hh"
#include "../detail/helpers.hh"
#include "../detail/access.hh"

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>

namespace Jetstream::Sakura {

struct ModalLayer::Impl {
    Config config;
    Box dim;
    Box window;
    Divider divider;
    Button close;
    Stack stack;
    Rect rect;
};

ModalLayer::ModalLayer() {
    impl = std::make_unique<Impl>();
    impl->divider.update({});
    impl->close.update({
        .str = "Close",
        .size = {Fill, 40.0f},
        .onClick = [impl = impl.get()]() {
            if (impl->config.onClose) {
                impl->config.onClose();
            }
        },
    });
}

ModalLayer::~ModalLayer() = default;

void ModalLayer::update(Config config) {
    if (config != impl->config) {
        invalidate(Dirty::Layout);
    }
    impl->config = std::move(config);
    if (!impl->config.content) {
        mountChildren({});
        Access::prepareInput(*this);
        return;
    }
    const bool fixedHeight = impl->config.size.y != 0.0f;
    impl->stack.update({
        .items = {
            {.child = impl->config.content, .grow = fixedHeight ? 1.0f : 0.0f},
            {.child = &impl->divider},
            {.child = &impl->close},
        },
    });
    mountChildren({&impl->dim, &impl->window, &impl->stack});
    Access::prepareInput(*this);
}

bool ModalLayer::modal() const {
    return impl->config.content != nullptr;
}

bool ModalLayer::covers(const Extent2D<F32>&) const {
    return impl->config.content != nullptr;
}

void ModalLayer::layout(const Context& ctx) {
    if (!impl->config.content) {
        return;
    }
    const Rect bounds = frame();
    const auto& config = impl->config;
    const F32 padding = Px(ctx, kWindowPadding);
    const F32 margin = Px(ctx, 19.0f);
    const F32 infinite = std::numeric_limits<F32>::infinity();
    const Extent2D<F32> room = {std::max(0.0f, bounds.width - 2.0f * margin), std::max(0.0f, bounds.height - 2.0f * margin)};
    F32 natural = 0.0f;
    if (config.size.x == 0.0f) {
        natural = measureChild(impl->stack, ctx, {infinite, infinite}).x + 2.0f * padding;
    }
    const F32 width = ResolveLength(ctx, config.size.x, natural, room.x);
    natural = 0.0f;
    if (config.size.y == 0.0f) {
        natural = measureChild(impl->stack, ctx, {std::max(0.0f, width - 2.0f * padding), infinite}).y + 2.0f * padding;
    }
    const F32 height = ResolveLength(ctx, config.size.y, natural, room.y);
    const Rect rect = {std::round(bounds.x + (bounds.width - width) * 0.5f),
                       std::round(bounds.y + (bounds.height - height) * 0.5f), std::ceil(width), std::ceil(height)};
    impl->rect = rect;

    impl->dim.update({.instances = {{.rect = bounds, .visible = true, .backgroundColor = ctx.color("modal_dim")}}});
    layoutChild(ctx, impl->dim, bounds);
    impl->window.update({
        .instances = {{.rect = rect, .visible = !rect.empty(), .backgroundColor = ctx.color("popup_bg")}},
        .cornerRadius = Px(ctx, kPanelRounding),
        .borderWidth = Px(ctx, 1.0f),
        .borderColor = ctx.color("border"),
        .strokeFringe = true,
    });
    layoutChild(ctx, impl->window, rect);
    layoutChild(ctx, impl->stack, Inset(rect, padding));
}

bool ModalLayer::mouseEvent(const MouseEvent& event) {
    if (!impl->config.content) {
        return false;
    }
    mouseEventChildren(event);
    return true;
}

bool ModalLayer::shortcutEvent(const KeyEvent& event) {
    if (!impl->config.content || event.type != KeyEventType::Press || event.key != KeyCode::Escape || event.repeat) {
        return false;
    }
    if (impl->config.onClose) {
        impl->config.onClose();
    }
    return true;
}

}  // namespace Jetstream::Sakura
