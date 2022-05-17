#ifndef JETSTREAM_RENDER_TYPES_HH
#define JETSTREAM_RENDER_TYPES_HH

#include "jetstream/types.hh"

namespace Jetstream::Render {

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

}  // namespace Jetstream::Render

#endif
