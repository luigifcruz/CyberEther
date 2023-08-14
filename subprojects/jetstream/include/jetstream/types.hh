#ifndef JETSTREAM_TYPE_HH
#define JETSTREAM_TYPE_HH

#include <any>
#include <span>
#include <vector>
#include <complex>
#include <typeindex>
#include <unordered_map>

#include "jetstream/memory/types.hh"

namespace Jetstream {

enum class Result : uint8_t {
    SUCCESS = 0,
    ERROR = 1,
    SKIP = 2,
    RECREATE = 3,
    CUDA_ERROR,
    VULKAN_ERROR,
    ASSERTION_ERROR,
    ERROR_TIMEOUT,
    ERROR_BEYOND_CAPACITY,
    CAST_ERROR,
};

template<typename T = U64>
struct Size2D {
    T width;
    T height;

    F32 ratio() const {
        return static_cast<F32>(width) / height;
    }

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

}  // namespace Jetstream

#endif
