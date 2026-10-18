#ifndef JETSTREAM_RENDER_SAKURA_DETAIL_HELPERS_HH
#define JETSTREAM_RENDER_SAKURA_DETAIL_HELPERS_HH

#include <jetstream/render/types.hh>
#include <jetstream/types.hh>

#include <algorithm>
#include <cmath>

namespace Jetstream::Sakura {

constexpr Rect Intersect(Rect a, Rect b) {
    const F32 x0 = std::max(a.x, b.x);
    const F32 y0 = std::max(a.y, b.y);
    const F32 x1 = std::min(a.right(), b.right());
    const F32 y1 = std::min(a.bottom(), b.bottom());
    return {x0, y0, std::max(0.0f, x1 - x0), std::max(0.0f, y1 - y0)};
}

constexpr Rect Inset(const Rect& rect, F32 inset) {
    return {rect.x + inset, rect.y + inset, std::max(0.0f, rect.width - 2.0f * inset),
            std::max(0.0f, rect.height - 2.0f * inset)};
}

constexpr Rect Inflate(const Rect& rect, F32 amount) {
    return {rect.x - amount, rect.y - amount, rect.width + 2.0f * amount, rect.height + 2.0f * amount};
}

inline Extent2D<F32> PixelToNdc(const Extent2D<U64>& framebufferSize, F32 x, F32 y) {
    if (framebufferSize.x == 0 || framebufferSize.y == 0) {
        return {-1.0f, 1.0f};
    }

    return {
        -1.0f + (2.0f * x / static_cast<F32>(framebufferSize.x)),
        1.0f - (2.0f * y / static_cast<F32>(framebufferSize.y)),
    };
}

inline Render::ScissorRect RectToScissor(const Rect& rect, const Extent2D<U64>& framebufferSize) {
    const F32 maxX = static_cast<F32>(framebufferSize.x);
    const F32 maxY = static_cast<F32>(framebufferSize.y);
    const F32 x = std::clamp(rect.x, 0.0f, maxX);
    const F32 y = std::clamp(rect.y, 0.0f, maxY);
    const F32 endX = std::clamp(rect.right(), x, maxX);
    const F32 endY = std::clamp(rect.bottom(), y, maxY);

    return {
        static_cast<U32>(std::floor(x)),
        static_cast<U32>(std::floor(y)),
        endX > x ? static_cast<U32>(std::ceil(endX) - std::floor(x)) : 0,
        endY > y ? static_cast<U32>(std::ceil(endY) - std::floor(y)) : 0,
    };
}

}  // namespace Jetstream::Sakura

#endif  // JETSTREAM_RENDER_SAKURA_DETAIL_HELPERS_HH
