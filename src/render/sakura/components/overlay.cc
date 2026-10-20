#include <jetstream/render/sakura/components/overlay.hh>

#include "../context.hh"
#include "../metrics.hh"
#include "../detail/anchor.hh"

#include <algorithm>
#include <utility>
#include <vector>

namespace Jetstream::Sakura {

struct Overlay::Impl {
    Config config;
};

Overlay::Overlay() {
    impl = std::make_unique<Impl>();
}

Overlay::~Overlay() = default;

void Overlay::update(Config config) {
    if (config != impl->config) {
        invalidate(Dirty::Layout);
    }
    std::vector<Component*> children;
    for (const auto& item : config.items) {
        if (item.child) {
            children.push_back(item.child);
        }
    }
    mountChildren(children);
    impl->config = std::move(config);
}

Extent2D<F32> Overlay::measure(const Context&, Extent2D<F32> available) {
    return {FillLength(available.x), FillLength(available.y)};
}

void Overlay::layout(const Context& ctx) {
    const Rect bounds = frame();
    for (const auto& item : impl->config.items) {
        if (!item.child || !item.child->visible()) {
            continue;
        }
        Extent2D<F32> size = {ResolveLength(ctx, item.size.x, 0.0f, bounds.width),
                              ResolveLength(ctx, item.size.y, 0.0f, bounds.height)};
        if (item.anchor != Anchor::Fill && (item.size.x == 0.0f || item.size.y == 0.0f)) {
            const auto measured = measureChild(*item.child, ctx, {item.size.x != 0.0f ? size.x : bounds.width,
                                                                  item.size.y != 0.0f ? size.y : bounds.height});
            size.x = item.size.x != 0.0f ? size.x : std::min(measured.x, bounds.width);
            size.y = item.size.y != 0.0f ? size.y : std::min(measured.y, bounds.height);
        }
        const Rect rect = Anchored(bounds, item.anchor, size, {Px(ctx, item.offset.x), Px(ctx, item.offset.y)}, 0.0f);
        layoutChild(ctx, *item.child, rect);
    }
}

}  // namespace Jetstream::Sakura
