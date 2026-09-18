#ifndef JETSTREAM_SUPERLUMINAL_SURFACE_INTERACTION_HH
#define JETSTREAM_SUPERLUMINAL_SURFACE_INTERACTION_HH

#include "render/surface_input.hh"

namespace Jetstream::detail {

// Called immediately after the surface's InvisibleButton so item state refers
// to the plot whose normalized coordinates are being forwarded.
inline void ForwardSuperluminalSurfaceInputEvents(const ImVec2& origin,
                                           const ImVec2& size,
                                           SurfaceInputState& state,
                                           const std::function<void(const InputEvent&)>& emit) {
    ForwardSurfaceInputEvents(origin, size, state, emit);
}

}  // namespace Jetstream::detail

#endif  // JETSTREAM_SUPERLUMINAL_SURFACE_INTERACTION_HH
