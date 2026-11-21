#include <jetstream/render/sakura/components/aside.hh>

#include "../context.hh"
#include "../metrics.hh"

#include <cmath>
#include <utility>
#include <vector>

namespace Jetstream::Sakura {

struct Aside::Impl {
    Config config;
};

Aside::Aside() {
    impl = std::make_unique<Impl>();
}

Aside::~Aside() = default;

void Aside::update(Config config) {
    if (config != impl->config) {
        invalidate(Dirty::Layout);
    }
    const bool remount = config.content != impl->config.content || config.rail != impl->config.rail ||
                         (config.content && !hasChild(*config.content)) || (config.rail && !hasChild(*config.rail));
    impl->config = std::move(config);
    if (remount) {
        std::vector<Component*> active;
        if (impl->config.content) {
            active.push_back(impl->config.content);
        }
        if (impl->config.rail) {
            active.push_back(impl->config.rail);
        }
        mountChildren(active);
    }
}

Extent2D<F32> Aside::measure(const Context&, Extent2D<F32> available) {
    return {FillLength(available.x), FillLength(available.y)};
}

void Aside::layout(const Context& ctx) {
    const auto& config = impl->config;
    const Rect bounds = frame();
    const F32 rail = std::round(Px(ctx, config.railWidth));
    const F32 inset = std::round(Px(ctx, config.railInset));
    if (config.content) {
        layoutChild(ctx, *config.content, bounds);
    }
    if (config.rail) {
        const bool shown = bounds.width >= rail + inset + Px(ctx, config.minimumContentWidth);
        if (config.rail->visible() && !shown) {
            cancelMouse(*config.rail);
        }
        config.rail->setVisible(shown);
        if (shown) {
            layoutChild(ctx, *config.rail, {bounds.right() - inset - rail, bounds.y, rail, bounds.height});
        }
    }
}

}  // namespace Jetstream::Sakura
