#include "jetstream/render/base/window_metrics.hh"

#include <algorithm>
#include <cmath>

namespace Jetstream::Render {

WindowMetrics WindowMetrics::Compute(const Extent2D<F32>& display,
                                     const Extent2D<F32>& framebuffer,
                                     const Extent2D<F32>& contentScale,
                                     F32 interfaceScale,
                                     Policy policy) {
    WindowMetrics metrics;
    metrics.displaySize = display;
    metrics.framebufferSize = framebuffer;

    if (display.x > 0.0f && display.y > 0.0f) {
        metrics.framebufferScale = {framebuffer.x / display.x, framebuffer.y / display.y};
    }

    metrics.scalingFactor = interfaceScale;
    if (policy == Policy::ContentScale && display.x == framebuffer.x && contentScale.x > 1.0f) {
        metrics.scalingFactor = interfaceScale * contentScale.x;
    }

    metrics.pixelRatio = metrics.scalingFactor * metrics.framebufferScale.x;

    return metrics;
}

Extent2D<U64> WindowMetrics::presentationSize() const {
    return {static_cast<U64>(displaySize.x * framebufferScale.x),
            static_cast<U64>(displaySize.y * framebufferScale.y)};
}

ScissorRect WindowMetrics::scissor(const Rect& clip) const {
    const auto limit = presentationSize();
    const F32 maxX = static_cast<F32>(limit.x);
    const F32 maxY = static_cast<F32>(limit.y);

    const F32 minX = std::max(0.0f, clip.x * framebufferScale.x);
    const F32 minY = std::max(0.0f, clip.y * framebufferScale.y);
    const F32 endX = std::min(maxX, clip.right() * framebufferScale.x);
    const F32 endY = std::min(maxY, clip.bottom() * framebufferScale.y);

    if (endX <= minX || endY <= minY) {
        return {};
    }

    const F32 x = std::floor(minX);
    const F32 y = std::floor(minY);
    return {
        static_cast<U32>(x),
        static_cast<U32>(y),
        static_cast<U32>(std::ceil(endX) - x),
        static_cast<U32>(std::ceil(endY) - y),
    };
}

}  // namespace Jetstream::Render
