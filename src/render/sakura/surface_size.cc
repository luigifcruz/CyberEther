#include <jetstream/render/sakura/surface_size.hh>

#include "base.hh"

namespace Jetstream::Sakura {

std::optional<SurfaceResize> ResolveSurfaceResize(const Context& ctx, const Extent2D<F32>& logicalSize) {
    if (logicalSize.x <= 0.0f || logicalSize.y <= 0.0f) {
        return std::nullopt;
    }

    const SurfaceResize resize{
        .logicalSize = {static_cast<U64>(logicalSize.x), static_cast<U64>(logicalSize.y)},
        .framebufferSize = LogicalFramebufferSize(ctx, logicalSize),
        .scale = SurfaceScale(ctx),
    };

    if (resize.logicalSize.x == 0 || resize.logicalSize.y == 0 ||
        resize.framebufferSize.x == 0 || resize.framebufferSize.y == 0) {
        return std::nullopt;
    }

    return resize;
}

}  // namespace Jetstream::Sakura
