#include <jetstream/render/sakura/components/scroll_area.hh>

#include <jetstream/render/sakura/components/box.hh>
#include <jetstream/render/sakura/components/scroller.hh>

#include "../context.hh"
#include "../metrics.hh"
#include "../detail/clip_layer.hh"

#include <algorithm>
#include <limits>
#include <utility>
#include <vector>

namespace Jetstream::Sakura {

struct ScrollArea::Impl {
    Config config;
    Box border;
    ClipLayer clipLayer;
    Scroller scroll;
    F32 scrollY = 0.0f;
    F32 contentHeight = 0.0f;
};

ScrollArea::ScrollArea() {
    impl = std::make_unique<Impl>();
    addChild(impl->scroll);
    addChild(impl->border);
    addChild(impl->clipLayer);
}

ScrollArea::~ScrollArea() = default;

void ScrollArea::update(Config config) {
    if (config != impl->config) {
        invalidate(Dirty::Layout);
    }
    std::vector<Component*> children;
    if (config.content) {
        children.push_back(config.content);
    }
    impl->clipLayer.mount(children);
    impl->config = std::move(config);
}

Extent2D<F32> ScrollArea::measure(const Context& ctx, Extent2D<F32> available) {
    const auto& config = impl->config;
    const F32 width = FillLength(available.x);
    F32 natural = 0.0f;
    if (config.height == 0.0f && config.content) {
        const F32 padding = config.border ? Px(ctx, kWindowPadding) : 0.0f;
        const F32 inner = std::max(0.0f, width - 2.0f * padding - (config.alwaysScrollbar ? Px(ctx, kScrollbarSize) : 0.0f));
        natural = measureChild(*config.content, ctx, {inner, std::numeric_limits<F32>::infinity()}).y + 2.0f * padding;
    }
    return {width, ResolveLength(ctx, config.height, natural, available.y)};
}

void ScrollArea::layout(const Context& ctx) {
    const Rect bounds = frame();
    const bool border = impl->config.border;
    const F32 padding = border ? Px(ctx, kWindowPadding) : 0.0f;
    const F32 infinite = std::numeric_limits<F32>::infinity();
    F32 scrollbar = Px(ctx, kScrollbarSize);
    F32 innerWidth = std::max(0.0f, bounds.width - 2.0f * padding - scrollbar);
    impl->contentHeight = 0.0f;
    if (impl->config.content) {
        if (!impl->config.alwaysScrollbar) {
            const F32 fullWidth = std::max(0.0f, bounds.width - 2.0f * padding);
            const F32 height = measureChild(*impl->config.content, ctx, {fullWidth, infinite}).y + 2.0f * padding;
            if (height <= bounds.height) {
                scrollbar = 0.0f;
                innerWidth = fullWidth;
            }
        }
        impl->contentHeight = measureChild(*impl->config.content, ctx, {innerWidth, infinite}).y + 2.0f * padding;
    }
    impl->scrollY = std::clamp(impl->scrollY, 0.0f, std::max(0.0f, impl->contentHeight - bounds.height));

    impl->border.update({
        .instances = {{.rect = bounds, .visible = border && !bounds.empty(), .backgroundColor = {0.0f, 0.0f, 0.0f, 0.0f}}},
        .clip = bounds,
        .borderWidth = Px(ctx, 1.0f),
        .borderColor = ctx.color("border"),
        .strokeFringe = true,
    });
    layoutChild(ctx, impl->border, bounds);

    impl->scroll.update({
        .contentHeight = impl->contentHeight,
        .scrollY = impl->scrollY,
        .wheelStep = Px(ctx, kWheelStep),
        .thickness = scrollbar,
        .margin = 0.0f,
        .thumbInset = Px(ctx, kScrollbarGrabInset),
        .trackColorKey = "scrollbar_bg",
        .thumbColorKey = "scrollbar_grab",
        .onScrollY = [this](F32 next) {
            impl->scrollY = next;
            invalidate(Dirty::Layout);
        },
    });
    layoutChild(ctx, impl->scroll, bounds);

    const F32 inset = padding * 0.5f;
    const Rect clipRect = {bounds.x + inset, bounds.y + inset,
                           std::max(0.0f, bounds.width - scrollbar - 2.0f * inset),
                           std::max(0.0f, bounds.height - 2.0f * inset)};
    layoutChild(ctx, impl->clipLayer, clipRect);
    if (impl->config.content) {
        impl->clipLayer.place(ctx, *impl->config.content,
                              {bounds.x + padding, bounds.y + padding - impl->scrollY, innerWidth,
                               std::max(0.0f, impl->contentHeight - 2.0f * padding)});
    }
}

}  // namespace Jetstream::Sakura
