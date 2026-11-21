#include "jetstream/render/base/present.hh"

#include "present_impl.hh"

#include <algorithm>
#include <utility>

namespace Jetstream::Render {

namespace {

void Push(PresentList& list, const WindowMetrics& metrics, const PresentList::Layer& layer, PresentList::Blend blend) {
    if (!layer.texture || layer.texture->raw() == 0 || layer.rect.empty()) {
        return;
    }
    list.sampled.insert(layer.texture.get());

    const ScissorRect scissor = metrics.scissor(layer.clip);
    if (scissor.width == 0 || scissor.height == 0) {
        return;
    }
    list.quads.push_back({
        .texture = layer.texture,
        .handle = layer.texture->raw(),
        .rect = layer.rect,
        .scissor = scissor,
        .blend = blend,
    });
}

}  // namespace

PresentList PresentList::Build(const WindowMetrics& metrics,
                               const std::vector<Layer>& underlays,
                               const std::optional<Layer>& canvas,
                               Blend canvasBlend) {
    PresentList list;
    for (const auto& underlay : underlays) {
        Push(list, metrics, underlay, Blend::Straight);
    }
    if (canvas.has_value()) {
        Push(list, metrics, *canvas, canvasBlend);
    }
    return list;
}

std::array<PresentGeometry::Vertex, 6> PresentGeometry::Triangles(const PresentList::Quad& quad,
                                                                  const Extent2D<F32>& display,
                                                                  bool yDown) {
    const F32 scaleX = 2.0f / display.x;
    const F32 scaleY = 2.0f / display.y;
    const auto x = [&](F32 value) {
        return value * scaleX - 1.0f;
    };
    const auto y = [&](F32 value) {
        return yDown ? value * scaleY - 1.0f : 1.0f - value * scaleY;
    };

    const Vertex topLeft = {x(quad.rect.x), y(quad.rect.y), 0.0f, 0.0f};
    const Vertex topRight = {x(quad.rect.right()), y(quad.rect.y), 1.0f, 0.0f};
    const Vertex bottomRight = {x(quad.rect.right()), y(quad.rect.bottom()), 1.0f, 1.0f};
    const Vertex bottomLeft = {x(quad.rect.x), y(quad.rect.bottom()), 0.0f, 1.0f};
    return {topLeft, topRight, bottomRight, topLeft, bottomRight, bottomLeft};
}

ScissorRect PresentGeometry::Clamp(const ScissorRect& scissor, const Extent2D<U64>& target) {
    const U64 x = std::min<U64>(scissor.x, target.x);
    const U64 y = std::min<U64>(scissor.y, target.y);
    return {
        static_cast<U32>(x),
        static_cast<U32>(y),
        static_cast<U32>(std::min<U64>(scissor.width, target.x - x)),
        static_cast<U32>(std::min<U64>(scissor.height, target.y - y)),
    };
}

void PresentList::append(PresentList other) {
    if (quads.empty() && sampled.empty()) {
        *this = std::move(other);
        return;
    }
    quads.insert(quads.end(), std::make_move_iterator(other.quads.begin()), std::make_move_iterator(other.quads.end()));
    sampled.insert(other.sampled.begin(), other.sampled.end());
}

void PresentList::clear() {
    quads.clear();
    sampled.clear();
}

bool PresentList::samples(const Texture& texture) const {
    return sampled.contains(&texture);
}

}  // namespace Jetstream::Render
