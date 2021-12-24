#ifndef RENDER_EXTRAS_H
#define RENDER_EXTRAS_H

#include <vector>

#include "render/base.hpp"
#include "render/type.hpp"

namespace Render::Extras {

static float a[] = {
    +1.0f, +1.0f, 0.0f,
    +1.0f, -1.0f, 0.0f,
    -1.0f, -1.0f, 0.0f,
    -1.0f, +1.0f, 0.0f,
};

static float b[] = {
    +1.0f, +1.0f,
    +1.0f, +0.0f,
    +0.0f, +0.0f,
    +0.0f, +1.0f,
};

inline std::vector<Render::Vertex::Buffer> FillScreenVertices() {
    return {
        {reinterpret_cast<float*>(&a), 12, 3},
        {reinterpret_cast<float*>(&b), 8, 2},
    };
}

inline std::vector<uint32_t> FillScreenIndices() {
    return {
        0, 1, 2,
        2, 3, 0,
    };
}

}  // namespace Render::Extras

#endif
