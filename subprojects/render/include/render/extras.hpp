#ifndef RENDER_EXTRAS_H
#define RENDER_EXTRAS_H

#include <vector>

#include "render/base.hpp"
#include "render/type.hpp"

namespace Render::Extras {

inline float FillScreenVertices[] = {
    +1.0f, +1.0f, 0.0f,
    +1.0f, -1.0f, 0.0f,
    -1.0f, -1.0f, 0.0f,
    -1.0f, +1.0f, 0.0f,
};

inline float FillScreenTextureVertices[] = {
    +1.0f, +1.0f,
    +1.0f, +0.0f,
    +0.0f, +0.0f,
    +0.0f, +1.0f,
};

inline uint32_t FillScreenIndices[] = {
    0, 1, 2,
    2, 3, 0,
};

}  // namespace Render::Extras

#endif
