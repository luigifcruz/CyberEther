#ifndef JETSTREAM_RENDER_TYPES_HH
#define JETSTREAM_RENDER_TYPES_HH

#include "jetstream/types.hh"

namespace Jetstream::Render {

enum class PixelFormat : uint32_t {
    RGBA,
    RED,
};

enum class DataFormat : uint32_t {
    RGBA,
    UI8,
    F32,
};

enum class PixelType : uint32_t {
    UI8,
    F32,
};

template<typename T>
struct Size2D {
    T width;
    T height;

    bool operator==(const Size2D<T>& a) const {
        return (width == a.width && height == a.height);
    }

    bool operator!=(const Size2D<T>& a) const {
        return (width != a.width || height != a.height);
    }

    bool operator<=(const Size2D<T>& a) const {
        return (width <= a.width || height <= a.height);
    }
};

inline Size2D<U64> NullSize = {0, 0};

}  // namespace Jetstream::Render

#endif
