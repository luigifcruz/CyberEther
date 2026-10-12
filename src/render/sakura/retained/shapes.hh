#ifndef JETSTREAM_RENDER_SAKURA_RETAINED_SHAPES_HH
#define JETSTREAM_RENDER_SAKURA_RETAINED_SHAPES_HH

#include <jetstream/render/sakura/components/retained/box.hh>

namespace Jetstream::Sakura::Retained {

struct CaretSize {
    F32 length;
    F32 depth;
};

inline Box::Instance Caret(const Rect& zone, bool down, const CaretSize& size,
                           const ColorRGBA<F32>& color, bool visible) {
    const auto center = zone.center();
    return {
        .rect = {center.x - size.length * 0.5f, center.y - size.depth * 0.5f, size.length, size.depth},
        .visible = visible,
        .backgroundColor = color,
        .rotation = down ? 180.0f : -90.0f,
    };
}

}  // namespace Jetstream::Sakura::Retained

#endif  // JETSTREAM_RENDER_SAKURA_RETAINED_SHAPES_HH
