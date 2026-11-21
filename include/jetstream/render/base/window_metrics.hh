#ifndef JETSTREAM_RENDER_BASE_WINDOW_METRICS_HH
#define JETSTREAM_RENDER_BASE_WINDOW_METRICS_HH

#include "jetstream/types.hh"
#include "jetstream/render/types.hh"

namespace Jetstream::Render {

struct JETSTREAM_API WindowMetrics {
    enum class Policy : U8 {
        Interface,
        ContentScale,
    };

    Extent2D<F32> displaySize = {0.0f, 0.0f};
    Extent2D<F32> framebufferSize = {0.0f, 0.0f};
    Extent2D<F32> framebufferScale = {1.0f, 1.0f};
    F32 scalingFactor = 1.0f;
    F32 pixelRatio = 1.0f;

    static WindowMetrics Compute(const Extent2D<F32>& display,
                                 const Extent2D<F32>& framebuffer,
                                 const Extent2D<F32>& contentScale,
                                 F32 interfaceScale,
                                 Policy policy);

    Extent2D<U64> presentationSize() const;
    ScissorRect scissor(const Rect& clip) const;

    bool operator==(const WindowMetrics&) const = default;
};

}  // namespace Jetstream::Render

#endif  // JETSTREAM_RENDER_BASE_WINDOW_METRICS_HH
