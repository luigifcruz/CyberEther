#include <jetstream/render/sakura/components/node/field_panel.hh>

#include <jetstream/render/sakura/components/box.hh>

#include "../../metrics.hh"

#include <cmath>
#include <limits>
#include <string>

namespace Jetstream::Sakura {

struct NodeFieldPanel::Impl {
    Config config;
    Box background;
};

NodeFieldPanel::NodeFieldPanel() {
    this->impl = std::make_unique<Impl>();
    setClipsChildren(true);
    addChild(impl->background);
}

NodeFieldPanel::~NodeFieldPanel() = default;

void NodeFieldPanel::update(Config config) {
    if (config != impl->config) {
        invalidate(Dirty::Layout);
    }
    impl->config = std::move(config);
    std::vector<Component*> children = {&impl->background};
    if (impl->config.content) {
        children.push_back(impl->config.content);
    }
    mountChildren(children);
}

Extent2D<F32> NodeFieldPanel::measure(const Context& ctx, Extent2D<F32> available) {
    const F32 width = FillLength(available.x);
    F32 natural = 0.0f;
    if (impl->config.height == 0.0f && impl->config.content) {
        const auto metrics = impl->config.content->metrics(ctx, width);
        natural = impl->config.content->hasFlexible() ? metrics.minimum : metrics.natural;
    }
    return {width, ResolveLength(ctx, impl->config.height, natural, available.y)};
}

void NodeFieldPanel::layout(const Context& ctx) {
    const Rect bounds = frame();
    const std::string backdrop = ctx.backdropColorKey.empty() ? "background" : ctx.backdropColorKey;
    impl->background.update({
        .instances = {{.rect = bounds, .visible = !bounds.empty(), .backgroundColor = ctx.color(backdrop)}},
    });
    layoutChild(ctx, impl->background, bounds);
    if (!impl->config.content) {
        return;
    }
    Context scoped = ctx;
    scoped.backdropColorKey = backdrop;
    layoutChild(scoped, *impl->config.content, bounds);
}

}  // namespace Jetstream::Sakura
