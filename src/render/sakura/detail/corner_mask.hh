#ifndef JETSTREAM_RENDER_SAKURA_DETAIL_CORNER_MASK_HH
#define JETSTREAM_RENDER_SAKURA_DETAIL_CORNER_MASK_HH

#include <jetstream/render/sakura/components/polyline.hh>

#include "../context.hh"
#include "helpers.hh"
#include "../metrics.hh"

#include <utility>
#include <vector>

namespace Jetstream::Sakura {

constexpr U64 kCornerMaskSegments = 8;
constexpr U64 kCornerMaskCapacity = 4 * (kCornerMaskSegments + 2) * 2;

inline void UpdateCornerMask(const Context& ctx, Polyline& mask, const Rect& rect, const Rect& clip) {
    std::vector<Polyline::Instance> instances;
    if (!ctx.backdropColorKey.empty() && !rect.empty()) {
        const auto bottom = ctx.color(ctx.backdropColorKey);
        const auto top = ctx.backdropTopColorKey.empty() ? bottom : ctx.color(ctx.backdropTopColorKey);
        instances = Polyline::CornerMasks(rect, kFrameRounding * ctx.pixelRatio, top, bottom, kCornerMaskSegments);
    }
    mask.update({
        .instances = std::move(instances),
        .clip = Intersect(rect, clip),
        .capacity = kCornerMaskCapacity,
    });
}

}  // namespace Jetstream::Sakura

#endif  // JETSTREAM_RENDER_SAKURA_DETAIL_CORNER_MASK_HH
