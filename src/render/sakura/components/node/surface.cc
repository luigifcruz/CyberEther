#include <jetstream/render/sakura/components/node/surface.hh>

#include <jetstream/render/base/texture.hh>
#include <jetstream/render/sakura/components/surface_image.hh>

#include "../../metrics.hh"

#include <memory>
#include <utility>

namespace Jetstream::Sakura {

struct NodeSurface::Impl {
    Config config;
    SurfaceImage surface;
};

NodeSurface::NodeSurface() {
    this->impl = std::make_shared<Impl>();
    addChild(impl->surface);
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
        .onInput = impl->config.onInput,
        .resolveCursor = impl->config.resolveCursor,
    });
}

Extent2D<F32> NodeSurface::measure(const Context& ctx, Extent2D<F32> available) {
    return {FillLength(available.x), FitLength(Px(ctx, impl->config.minimumHeight), available.y)};
}

void NodeSurface::layout(const Context& ctx) {
    layoutChild(ctx, impl->surface, frame());
}

}  // namespace Jetstream::Sakura
