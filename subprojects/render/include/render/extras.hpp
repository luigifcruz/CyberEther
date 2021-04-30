#ifndef RENDER_EXTRAS_H
#define RENDER_EXTRAS_H

#include "render/base.hpp"
#include "render/types.hpp"

namespace Render::Extras {

static float a[] = {
    +1.0f, +1.0f, 0.0f,
    +1.0f, -1.0f, 0.0f,
    -1.0f, -1.0f, 0.0f,
    -1.0f, +1.0f, 0.0f,
};

static float b[] = {
    +0.0f, +0.0f,
    +0.0f, +1.0f,
    +1.0f, +1.0f,
    +1.0f, +0.0f,
};

static std::vector<Render::Vertex::Buffer> FillScreenVertices = {
    {(float*)&a, 12, 3},
    {(float*)&b, 8, 2},
};

static std::vector<uint> FillScreenIndices = {
    0, 1, 2,
    2, 3, 0,
};

} // namespace Render::Extras

#endif
