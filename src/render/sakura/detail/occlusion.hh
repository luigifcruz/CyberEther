#ifndef JETSTREAM_RENDER_SAKURA_DETAIL_OCCLUSION_HH
#define JETSTREAM_RENDER_SAKURA_DETAIL_OCCLUSION_HH

#include <jetstream/types.hh>

#include "helpers.hh"

#include <algorithm>
#include <optional>
#include <vector>

namespace Jetstream::Sakura {

constexpr F32 kOcclusionMinimumArea = 64.0f * 64.0f;
constexpr F32 kOcclusionEdgeMargin = 1.0f;

inline F32 RectArea(const Rect& rect) {
    return rect.empty() ? 0.0f : rect.width * rect.height;
}

inline Rect HollowInterior(const Rect& shape,
                           F32 cornerRadius,
                           F32 borderWidth,
                           F32 softness,
                           const ColorRGBA<F32>& fill) {
    if (softness > 0.0f || borderWidth <= 0.0f || fill.a > 0.0f) {
        return {};
    }
    return Inset(shape, std::max(cornerRadius, 2.0f * borderWidth + kOcclusionEdgeMargin));
}

inline Rect OpaqueInterior(const Rect& shape,
                           F32 cornerRadius,
                           F32 borderWidth,
                           F32 softness,
                           const ColorRGBA<F32>& fill,
                           const std::optional<Rect>& clip) {
    if (softness > 0.0f || fill.a < 1.0f) {
        return {};
    }
    Rect interior = Inset(shape, std::max(cornerRadius, borderWidth) + kOcclusionEdgeMargin);
    if (clip.has_value()) {
        interior = Intersect(interior, *clip);
    }
    return RectArea(interior) >= kOcclusionMinimumArea ? interior : Rect{};
}

inline Rect LargestHole(const Rect& shape, const Rect& own, const std::vector<Rect>& occluders) {
    Rect best = Intersect(shape, own);
    for (const auto& occluder : occluders) {
        const Rect covered = Intersect(shape, occluder);
        if (RectArea(covered) > RectArea(best)) {
            best = covered;
        }
    }
    return RectArea(best) >= kOcclusionMinimumArea ? best : Rect{};
}

inline Rect LocalHole(const Rect& shape, const Rect& hole) {
    if (shape.empty() || hole.empty()) {
        return {};
    }
    const auto center = shape.center();
    return {
        (hole.x - center.x) / shape.width,
        (center.y - hole.bottom()) / shape.height,
        hole.width / shape.width,
        hole.height / shape.height,
    };
}

}  // namespace Jetstream::Sakura

#endif  // JETSTREAM_RENDER_SAKURA_DETAIL_OCCLUSION_HH
