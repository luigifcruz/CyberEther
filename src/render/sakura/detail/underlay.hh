#ifndef JETSTREAM_RENDER_SAKURA_DETAIL_UNDERLAY_HH
#define JETSTREAM_RENDER_SAKURA_DETAIL_UNDERLAY_HH

#include <jetstream/render/base/texture.hh>
#include <jetstream/render/types.hh>
#include <jetstream/types.hh>

#include "helpers.hh"

#include <memory>
#include <optional>
#include <vector>

namespace Jetstream::Sakura {

struct Underlay {
    std::shared_ptr<const Render::Texture> texture;
    Rect rect;
    Rect clip;
    Extent2D<U64> frame;
};

struct LiveRegion {
    U64 order = 0;
    Rect rect;
};

inline bool UnderlaysAvailable(bool eraseBlending, const ColorRGBA<F32>& clearColor) {
    return eraseBlending && clearColor.a >= 1.0f;
}

inline std::optional<Underlay> MakeUnderlay(const std::shared_ptr<const Render::Texture>& texture,
                                            const Extent2D<F32>& translation,
                                            const Rect& rect,
                                            const Render::ScissorRect& scissor,
                                            const Extent2D<U64>& frame) {
    if (!texture || rect.empty() || scissor.width == 0 || scissor.height == 0) {
        return std::nullopt;
    }
    return Underlay{
        .texture = texture,
        .rect = {translation.x + rect.x, translation.y + rect.y, rect.width, rect.height},
        .clip = {static_cast<F32>(scissor.x), static_cast<F32>(scissor.y), static_cast<F32>(scissor.width),
                 static_cast<F32>(scissor.height)},
        .frame = frame,
    };
}

inline bool OpaqueCovers(const Rect& shape,
                         F32 cornerRadius,
                         const ColorRGBA<F32>& color,
                         const std::optional<Rect>& clip,
                         const Rect& target) {
    if (color.a < 1.0f || shape.empty() || target.empty()) {
        return false;
    }
    const F32 radius = std::max(0.0f, cornerRadius);
    const Rect body = {shape.x + radius, shape.y + radius, shape.width - 2.0f * radius, shape.height - 2.0f * radius};
    const auto inside = [&](const Rect& outer) {
        return target.x >= outer.x && target.y >= outer.y && target.right() <= outer.right() &&
               target.bottom() <= outer.bottom();
    };
    return inside(body) && (!clip.has_value() || inside(*clip));
}

template<typename Covered>
std::vector<bool> UnderlayEligibility(const std::vector<LiveRegion>& regions, Covered&& covered) {
    std::vector<bool> allowed(regions.size(), true);
    for (U64 later = 0; later < regions.size(); ++later) {
        for (U64 earlier = 0; earlier < later && allowed[later]; ++earlier) {
            if (!allowed[earlier]) {
                continue;
            }
            const Rect overlap = Intersect(regions[earlier].rect, regions[later].rect);
            if (!overlap.empty() && !covered(regions[earlier].order, regions[later].order, overlap)) {
                allowed[later] = false;
            }
        }
    }
    return allowed;
}

inline Rect MapToScreen(const Rect& rect, const Rect& screen, const Extent2D<U64>& framebuffer) {
    if (framebuffer.x == 0 || framebuffer.y == 0) {
        return {};
    }
    const F32 scaleX = screen.width / static_cast<F32>(framebuffer.x);
    const F32 scaleY = screen.height / static_cast<F32>(framebuffer.y);
    return {screen.x + rect.x * scaleX, screen.y + rect.y * scaleY, rect.width * scaleX, rect.height * scaleY};
}

}  // namespace Jetstream::Sakura

#endif  // JETSTREAM_RENDER_SAKURA_DETAIL_UNDERLAY_HH
