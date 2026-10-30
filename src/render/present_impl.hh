#ifndef JETSTREAM_RENDER_PRESENT_IMPL_HH
#define JETSTREAM_RENDER_PRESENT_IMPL_HH

#include "jetstream/render/base/present.hh"

#include <array>

namespace Jetstream::Render {

struct JETSTREAM_API PresentGeometry {
    struct Vertex {
        F32 x = 0.0f;
        F32 y = 0.0f;
        F32 u = 0.0f;
        F32 v = 0.0f;
    };

    static std::array<Vertex, 6> Triangles(const PresentList::Quad& quad, const Extent2D<F32>& display, bool yDown);
    static ScissorRect Clamp(const ScissorRect& scissor, const Extent2D<U64>& target);
};

}  // namespace Jetstream::Render

#endif  // JETSTREAM_RENDER_PRESENT_IMPL_HH
