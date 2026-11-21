#include <jetstream/render/sakura/components/divider.hh>

#include <jetstream/render/sakura/components/box.hh>

#include "../context.hh"
#include "../metrics.hh"

#include <algorithm>
#include <cmath>
#include <utility>

namespace Jetstream::Sakura {

struct Divider::Impl {
    Config config;
    Box line;

    F32 spacing() const {
        return std::max(0.0f, config.spacing);
    }
};

Divider::Divider() {
    impl = std::make_unique<Impl>();
    addChild(impl->line);
}

Divider::~Divider() = default;

void Divider::update(Config config) {
    if (config == impl->config) {
        impl->config = std::move(config);
        return;
    }
    invalidate(Dirty::Layout);
    impl->config = config;
}

Extent2D<F32> Divider::measure(const Context& ctx, Extent2D<F32> available) {
    const F32 thickness = impl->config.separator ? Hairline(ctx) : 0.0f;
    return {FillLength(available.x), FitLength(2.0f * Px(ctx, impl->spacing()) + thickness, available.y)};
}

void Divider::layout(const Context& ctx) {
    const Rect bounds = frame();
    const F32 thickness = Hairline(ctx);
    const Rect rect = {bounds.x, std::round(bounds.y + Px(ctx, impl->spacing())), bounds.width, thickness};
    impl->line.update({
        .instances = {{.rect = rect, .visible = impl->config.separator && !bounds.empty(),
                       .backgroundColor = ctx.color("separator")}},
        .clip = bounds,
    });
    layoutChild(ctx, impl->line, bounds);
}

}  // namespace Jetstream::Sakura
