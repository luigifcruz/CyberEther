#ifndef JETSTREAM_TYPE_HH
#define JETSTREAM_TYPE_HH
#define _USE_MATH_DEFINES

#include <any>
#include <span>
#include <vector>
#include <complex>
#include <typeindex>
#include <unordered_map>

#include "jetstream/memory/types.hh"

#ifdef _WIN32
#undef max
#undef min
#undef ERROR
#undef pow10
#undef read
#undef _read
#endif

namespace Jetstream {

enum class Result : uint8_t {
    SUCCESS     = 0,
    ERROR       = 1,
    FATAL       = 2,
    SKIP        = 3,
    RECREATE    = 4,
    TIMEOUT     = 5,
};

inline constexpr Result& operator|=(Result& lhs, const Result& rhs) {
    lhs = ((lhs != Result::SUCCESS) || (rhs != Result::SUCCESS)) ? Result::ERROR : Result::SUCCESS;
    return lhs;
}

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
