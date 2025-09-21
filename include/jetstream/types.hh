#ifndef JETSTREAM_TYPE_HH
#define JETSTREAM_TYPE_HH

#include <any>
#include <vector>
#include <complex>
#include <typeindex>
#include <unordered_map>

#include "jetstream/memory/types.hh"

namespace Jetstream {

enum class Result : uint16_t {
    SUCCESS     = 0,
    ERROR       = 1,
    WARNING     = 2,
    FATAL       = 3,
    SKIP        = 4,
    YIELD       = 5,
    RELOAD      = 6,
    RECREATE    = 7,
    TIMEOUT     = 8,
};

//
// Taint
//

enum class Taint : uint64_t {
    CLEAN               = 0 << 0, ///< No taint set, data is in its original state.
    IN_PLACE            = 1 << 0, ///< Module will overwrite input, modifying it directly.
    DISCONTIGUOUS       = 1 << 1, ///< Accepts non-contiguous data buffers for input tensors.
};

inline Taint operator&(const Taint& lhs, const Taint& rhs) {
    return static_cast<Taint>(reinterpret_cast<const uint64_t&>(lhs) & reinterpret_cast<const uint64_t&>(rhs));
}

//
// ColorRGBA
//

template<typename T = F32>
struct ColorRGBA {
    T r;
    T g;
    T b;
    T a;

    bool operator==(const ColorRGBA<T>& c) const {
        return (r == c.r && g == c.g && b == c.b && a == c.a);
    }

    bool operator!=(const ColorRGBA<T>& c) const {
        return (r != c.r || g != c.g || b != c.b || a != c.a);
    }
};

//
// Extent2D
//

template<typename T = U64>
struct Extent2D {
    T x;
    T y;

    F32 ratio() const {
        return static_cast<F32>(x) / y;
    }

    bool operator==(const Extent2D<T>& a) const {
        return (x == a.x && y == a.y);
    }

    bool operator!=(const Extent2D<T>& a) const {
        return (x != a.x || y != a.y);
    }

    bool operator<=(const Extent2D<T>& a) const {
        return (x <= a.x || y <= a.y);
    }
};

template<typename T>
Extent2D<T> operator*(const Extent2D<T>& a, const F32& b) {
    return {static_cast<T>(a.x * b), static_cast<T>(a.y * b)};
}

template<typename T>
Extent2D<T> operator/(const Extent2D<T>& a, const F32& b) {
    return {static_cast<T>(a.x / b), static_cast<T>(a.y / b)};
}

inline Extent2D<U64> NullSize2D = {0, 0};

//
// Extent3D
//

template<typename T = U64>
struct Extent3D {
    T x;
    T y;
    T z;

    F32 ratio() const {
        return static_cast<F32>(x) / y;
    }

    bool operator==(const Extent3D<T>& a) const {
        return (x == a.x && y == a.y && z == a.z);
    }

    bool operator!=(const Extent3D<T>& a) const {
        return (x != a.x || y != a.y || z != a.z);
    }

    bool operator<=(const Extent3D<T>& a) const {
        return (x <= a.x || y <= a.y || z <= a.z);
    }
};

template<typename T>
Extent3D<T> operator*(const Extent3D<T>& a, const F32& b) {
    return {static_cast<T>(a.x * b), static_cast<T>(a.y * b), static_cast<T>(a.z * b)};
}

template<typename T>
Extent3D<T> operator/(const Extent3D<T>& a, const F32& b) {
    return {static_cast<T>(a.x / b), static_cast<T>(a.y / b), static_cast<T>(a.z / b)};
}

inline Extent3D<U64> NullSize3D = {0, 0, 0};

//
// Extent4D
//

template<typename T = U64>
struct Extent4D {
    T x;
    T y;
    T z;
    T w;

    F32 ratio() const {
        return static_cast<F32>(x) / y;
    }

    bool operator==(const Extent4D<T>& a) const {
        return (x == a.x && y == a.y && z == a.z && w == a.w);
    }

    bool operator!=(const Extent4D<T>& a) const {
        return (x != a.x || y != a.y || z != a.z || w != a.w);
    }

    bool operator<=(const Extent4D<T>& a) const {
        return (x <= a.x || y <= a.y || z <= a.z || w <= a.w);
    }
};

template<typename T>
Extent4D<T> operator*(const Extent4D<T>& a, const F32& b) {
    return {static_cast<T>(a.x * b), static_cast<T>(a.y * b), static_cast<T>(a.z * b), static_cast<T>(a.w * b)};
}

template<typename T>
Extent4D<T> operator/(const Extent4D<T>& a, const F32& b) {
    return {static_cast<T>(a.x / b), static_cast<T>(a.y / b), static_cast<T>(a.z / b), static_cast<T>(a.w / b)};
}

inline Extent4D<U64> NullSize4D = {0, 0, 0, 0};

}  // namespace Jetstream

#endif
