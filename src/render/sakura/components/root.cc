#include <jetstream/render/sakura/components/root.hh>

#include <jetstream/render/sakura/components/canvas.hh>

#include "../context.hh"
#include "../metrics.hh"
#include "../detail/root_component.hh"

#include <utility>

namespace Jetstream::Sakura {

struct Root::Impl {
    Config config;
    Canvas canvas;
    RootComponent root;

    Impl() {
        canvas.mount(root);
    }
};

Root::Root() {
    impl = std::make_unique<Impl>();
}

Root::~Root() = default;

void Root::update(Config config) {
    impl->config = std::move(config);
    impl->root.setLayers(impl->config.layers);
    impl->canvas.update({.id = impl->config.id + ":root"});
}

void Root::render(const Sakura::Context& ctx) {
    impl->canvas.render(ctx, DisplayMetrics::RootRegion(Metrics(ctx), impl->config.topOffset));
}

}  // namespace Jetstream::Sakura
