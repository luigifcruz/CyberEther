#ifndef JETSTREAM_TYPE_HH
#define JETSTREAM_TYPE_HH

#include <span>
#include <complex>

namespace Jetstream {

enum class Result : uint8_t {
    SUCCESS = 0,
    ERROR = 1,
    CUDA_ERROR,
    ASSERTION_ERROR,
};

template<typename T>
struct Range {
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
};

}  // namespace Jetstream

#endif
