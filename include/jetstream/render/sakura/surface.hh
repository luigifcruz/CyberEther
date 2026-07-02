#ifndef JETSTREAM_RENDER_SAKURA_SURFACE_HH
#define JETSTREAM_RENDER_SAKURA_SURFACE_HH

#include <jetstream/types.hh>

namespace Jetstream::Sakura {

struct SurfaceResize {
    Extent2D<U64> logicalSize = {0, 0};
    Extent2D<U64> framebufferSize = {0, 0};
    F32 scale = 1.0f;
};

}  // namespace Jetstream::Sakura

#endif  // JETSTREAM_RENDER_SAKURA_SURFACE_HH
