#ifndef JETSTREAM_TYPE_HH
#define JETSTREAM_TYPE_HH

#include <any>
#include <vector>
#include <complex>
#include <typeindex>
#include <unordered_map>

#include "jetstream/memory/types.hh"

namespace Jetstream {

//
// Result
//

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
    INCOMPLETE  = 9,
};

inline std::ostream& operator<<(std::ostream& os, const Result& result) {
    switch (result) {
        case Result::SUCCESS:    return os << "SUCCESS";
        case Result::ERROR:      return os << "ERROR";
        case Result::WARNING:    return os << "WARNING";
        case Result::FATAL:      return os << "FATAL";
        case Result::SKIP:       return os << "SKIP";
        case Result::YIELD:      return os << "YIELD";
        case Result::RELOAD:     return os << "RELOAD";
        case Result::RECREATE:   return os << "RECREATE";
        case Result::TIMEOUT:    return os << "TIMEOUT";
        case Result::INCOMPLETE: return os << "INCOMPLETE";
        default:                 return os << "UNKNOWN";
    }
}

//
// Common Numeric Constants
//

#ifndef JST_PI
#define JST_PI 3.14159265358979323846
#endif

#ifndef JST_PI_2
#define JST_PI_2 1.57079632679489661923
#endif

#ifndef JST_E
#define JST_E 2.7182818284590452354
#endif

#ifndef JST_MB
#define JST_MB (1024*1024)
#endif

#ifndef JST_MHZ
#define JST_MHZ (1000*1000)
#endif

//
// Range
//

template<typename T>
struct JETSTREAM_API Range {
    T min;
    T max;

    bool operator==(const Range<T>& a) const {
        return (min == a.min && max == a.max);
    }

    bool operator!=(const Range<T>& a) const {
        return (min != a.min || max != a.max);
    }

    bool operator<=(const Range<T>& a) const {
        return (min <= a.min || max <= a.max);
    }

    friend inline std::ostream& operator<<(std::ostream& os, const Range<T>& range) {
        return os << jst::fmt::format("[{}, {}]", range.min, range.max);
    }
};

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

template <> struct jst::fmt::formatter<Jetstream::Result> : ostream_formatter {};

#endif
