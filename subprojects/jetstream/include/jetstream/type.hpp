#ifndef JETSTREAM_TYPE_H
#define JETSTREAM_TYPE_H

#include <iostream>
#include <memory>
#include <condition_variable>
#include <mutex>
#include <thread>
#include <atomic>
#include <vector>
#include <complex>

#include "jetstream/tools/span.hpp"

#if __has_include("nvtx3/nvToolsExt.h") && defined JETSTREAM_DEBUG
#include <nvtx3/nvToolsExt.h>

#ifndef DEBUG_PUSH
#define DEBUG_PUSH(name) { nvtxRangePushA(name); }
#endif
#ifndef DEBUG_POP
#define DEBUG_POP() { nvtxRangePop(); }
#endif

#else

#ifndef DEBUG_PUSH
#define DEBUG_PUSH(name)
#endif
#ifndef DEBUG_POP
#define DEBUG_POP()
#endif

#endif

namespace Jetstream {

enum class Locale : uint8_t {
    CPU     = 1 << 0,
    CUDA    = 1 << 1,
};

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
