#ifndef JETSTREAM_ASSETS_CONSTANTS_HH
#define JETSTREAM_ASSETS_CONSTANTS_HH

#include <cstdint>

inline float FillScreenVertices[] = {
    +1.0f, -1.0f, 0.0f,
    +1.0f, +1.0f, 0.0f,
    -1.0f, +1.0f, 0.0f,
    -1.0f, -1.0f, 0.0f,
};

inline float FillScreenTextureVertices[] = {
    +1.0f, +0.0f,
    +1.0f, +1.0f,
    +0.0f, +1.0f,
    +0.0f, +0.0f,
};

inline float FillScreenTextureVerticesXYFlip[] = {
    +1.0f, +1.0f,
    +1.0f, +0.0f,
    +0.0f, +0.0f,
    +0.0f, +1.0f,
};

inline uint32_t FillScreenIndices[] = {
    0, 1, 2,
    2, 3, 0,
};

#endif
