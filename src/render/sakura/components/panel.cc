#include <jetstream/render/sakura/components/panel.hh>

#include <jetstream/render/sakura/components/box.hh>

#include "../context.hh"
#include "../metrics.hh"

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>
#include <vector>

namespace Jetstream::Sakura {

struct Panel::Impl {
    Config config;
    Box background;
};

Panel::Panel() {
    impl = std::make_unique<Impl>();
    setClipsChildren(true);
    addChild(impl->background);
}

Panel::~Panel() = default;

void Panel::update(Config config) {
    if (config != impl->config) {
        invalidate(Dirty::Layout);
    }
    const bool remount = config.content != impl->config.content || !hasChild(impl->background) ||
                         (config.content && !hasChild(*config.content));
    impl->config = std::move(config);
    if (!remount) {
        return;
    }
    std::vector<Component*> children = {&impl->background};
    if (impl->config.content) {
        children.push_back(impl->config.content);
    }
    mountChildren(children);
}

Extent2D<F32> Panel::measure(const Context& ctx, Extent2D<F32> available) {
    const auto& config = impl->config;
    const F32 infinite = std::numeric_limits<F32>::infinity();
    F32 natural = 0.0f;
    if (config.size.x == 0.0f && config.content) {
        natural = std::ceil(measureChild(*config.content, ctx, {infinite, infinite}).x);
    }
    F32 width = ResolveLength(ctx, config.size.x, natural, available.x);
    if (config.maxWidth > 0.0f) {
        width = std::min(width, Px(ctx, config.maxWidth));
    }
    natural = 0.0f;
    if (config.size.y == 0.0f && config.content) {
        natural = measureChild(*config.content, ctx, {width, infinite}).y;
    }
    return {width, ResolveLength(ctx, config.size.y, natural, available.y)};
}

void Panel::layout(const Context& ctx) {
    const Rect bounds = frame();
    const auto& config = impl->config;
    impl->background.update({
        .instances = {{.rect = bounds, .visible = !config.transparent && !bounds.empty(),
                       .backgroundColor = ctx.color(config.backgroundColorKey)}},
    });
    layoutChild(ctx, impl->background, bounds);
    if (config.content) {
        layoutChild(ctx, *config.content, bounds);
    }
}

}  // namespace Jetstream::Sakura
