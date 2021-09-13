#ifndef JETSTREAM_TYPE_H
#define JETSTREAM_TYPE_H

#include <iostream>
#include <memory>
#include <condition_variable>
#include <mutex>
#include <thread>
#include <vector>
#include <complex>

#include "jetstream/tools/span.hpp"
#include "jetstream/helpers.hpp"

namespace Jetstream {

enum class Locale : uint8_t {
    CPU     = 1 << 0,
    CUDA    = 1 << 1,
};

inline Locale operator&(Locale lhs, Locale rhs) {
    return static_cast<Locale>(static_cast<uint8_t>(lhs) & static_cast<uint8_t>(rhs));
}

inline Locale operator|(Locale lhs, Locale rhs) {
    return static_cast<Locale>(static_cast<uint8_t>(lhs) | static_cast<uint8_t>(rhs));
}

template<typename T>
struct Data {
    Locale location;
    T buf;
};

enum Result {
    SUCCESS = 0,
    ERROR = 1,
};

typedef nonstd::span<float> VF32;
typedef nonstd::span<std::complex<float>> VCF32;

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

template<typename T>
struct Range {
    T min;
    T max;

    bool operator==(const Size2D<T>& a) const {
        return (min == a.min && max == a.max);
    }

    bool operator!=(const Size2D<T>& a) const {
        return (min != a.min || max != a.max);
    }

    bool operator<=(const Size2D<T>& a) const {
        return (min <= a.min || max <= a.max);
    }
};

} // namespace Jetstream

#endif
