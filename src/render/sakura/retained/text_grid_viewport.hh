#ifndef JETSTREAM_RENDER_SAKURA_RETAINED_TEXT_GRID_VIEWPORT_HH
#define JETSTREAM_RENDER_SAKURA_RETAINED_TEXT_GRID_VIEWPORT_HH

#include "helpers.hh"

#include <algorithm>
#include <cmath>

namespace Jetstream::Sakura::Retained {

struct TextGridViewport {
    Rect bounds;
    U64 rowCapacity = 0;

    void update(const Rect& frame, const Rect& clip, F32 minimumRowHeight,
                U64 columns, U64 minimumCapacity) {
        bounds = Intersect(frame, clip);

        constexpr U64 step = 16;
        const U64 rows = static_cast<U64>(std::ceil(bounds.height / minimumRowHeight)) + 1;
        const U64 groupedRows = rows * columns;
        const U64 required = ((groupedRows + step - 1) / step) * step;

        rowCapacity = std::max({rowCapacity, minimumCapacity, required});
    }

    U64 poolCapacity(U64 current, U64 demand, U64 maxSegments) const {
        const U64 step = std::max<U64>(1, rowCapacity);
        const U64 required = ((demand + step - 1) / step) * step;
        return std::min(step * std::max<U64>(1, maxSegments), std::max({current, step, required}));
    }
};

}  // namespace Jetstream::Sakura::Retained

#endif  // JETSTREAM_RENDER_SAKURA_RETAINED_TEXT_GRID_VIEWPORT_HH
