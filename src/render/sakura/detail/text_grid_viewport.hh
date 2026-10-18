#ifndef JETSTREAM_RENDER_SAKURA_DETAIL_TEXT_GRID_VIEWPORT_HH
#define JETSTREAM_RENDER_SAKURA_DETAIL_TEXT_GRID_VIEWPORT_HH

#include "helpers.hh"

#include <algorithm>
#include <cmath>

namespace Jetstream::Sakura {

struct TextGridViewport {
    Rect bounds;
    U64 rowCapacity = 0;

    void fixed(const Rect& frame, const Rect& clip, U64 rows) {
        bounds = Intersect(frame, clip);
        rowCapacity = rows;
    }

    void update(const Rect& frame, const Rect& clip, F32 rowHeight, U64 minimumCapacity, F32 ceilingHeight) {
        bounds = Intersect(frame, clip);
        rowCapacity = std::max(rowCapacity, minimumCapacity);
        const U64 required = RowsFor(bounds.height, rowHeight);
        if (required > rowCapacity) {
            rowCapacity = std::max(required, RowsFor(ceilingHeight, rowHeight));
        }
    }

 private:
    static U64 RowsFor(F32 height, F32 rowHeight) {
        constexpr U64 step = 16;
        const U64 rows = static_cast<U64>(std::ceil(std::max(0.0f, height) / std::max(1.0f, rowHeight))) + 1;
        return ((rows + step - 1) / step) * step;
    }
};

}  // namespace Jetstream::Sakura

#endif  // JETSTREAM_RENDER_SAKURA_DETAIL_TEXT_GRID_VIEWPORT_HH
