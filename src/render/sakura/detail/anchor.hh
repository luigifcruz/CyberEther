#ifndef JETSTREAM_RENDER_SAKURA_DETAIL_ANCHOR_HH
#define JETSTREAM_RENDER_SAKURA_DETAIL_ANCHOR_HH

#include <jetstream/render/sakura/components/overlay.hh>
#include <jetstream/types.hh>

#include <cmath>

namespace Jetstream::Sakura {

inline Rect Anchored(const Rect& bounds, Anchor anchor, Extent2D<F32> size, Extent2D<F32> offset, F32 margin) {
    switch (anchor) {
        case Anchor::Fill:
            return bounds;
        case Anchor::TopLeft:
            return {bounds.x + margin + offset.x, bounds.y + margin + offset.y, size.x, size.y};
        case Anchor::TopCenter:
            return {bounds.x + std::floor((bounds.width - size.x) * 0.5f) + offset.x, bounds.y + margin + offset.y,
                    size.x, size.y};
        case Anchor::TopRight:
            return {bounds.right() - margin - size.x + offset.x, bounds.y + margin + offset.y, size.x, size.y};
        case Anchor::Center:
            return {bounds.x + std::floor((bounds.width - size.x) * 0.5f) + offset.x,
                    bounds.y + std::floor((bounds.height - size.y) * 0.5f) + offset.y, size.x, size.y};
        case Anchor::BottomLeft:
            return {bounds.x + margin + offset.x, bounds.bottom() - margin - size.y + offset.y, size.x, size.y};
        case Anchor::BottomCenter:
            return {bounds.x + std::floor((bounds.width - size.x) * 0.5f) + offset.x,
                    bounds.bottom() - margin - size.y + offset.y, size.x, size.y};
        case Anchor::BottomRight:
            return {bounds.right() - margin - size.x + offset.x, bounds.bottom() - margin - size.y + offset.y,
                    size.x, size.y};
    }
    return bounds;
}

}  // namespace Jetstream::Sakura

#endif  // JETSTREAM_RENDER_SAKURA_DETAIL_ANCHOR_HH
