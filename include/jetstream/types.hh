#ifndef JETSTREAM_TYPE_HH
#define JETSTREAM_TYPE_HH

#include <any>
#include <vector>
#include <complex>
#include <typeindex>
#include <unordered_map>

#include "jetstream/memory/types.hh"

namespace Jetstream {

enum class Result : uint8_t {
    SUCCESS     = 0,
    ERROR       = 1,
    WARNING     = 2,
    FATAL       = 3,
    SKIP        = 4,
    RELOAD      = 5,
    RECREATE    = 6,
    TIMEOUT     = 7,
};

inline constexpr Result& operator|=(Result& lhs, const Result& rhs) {
    lhs = ((lhs != Result::SUCCESS) || (rhs != Result::SUCCESS)) ? Result::ERROR : Result::SUCCESS;
    return lhs;
}

/**
 * @enum Taint
 * @brief Represents the taint status of a module's input or output.
 *
 * This enum class is used to describe various taint or state conditions of data being processed by a module. 
 */
enum class Taint : uint64_t {
    CLEAN               = 0 << 0, ///< No taint set, data is in its original state.
    IN_PLACE            = 1 << 0, ///< Module will overwrite input, modifying it directly.
    DISCONTIGUOUS       = 1 << 1, ///< Accepts non-contiguous data buffers for input tensors.
};

inline constexpr Taint operator&(const Taint& lhs, const Taint& rhs) {
    return static_cast<Taint>(reinterpret_cast<const uint64_t&>(lhs) & reinterpret_cast<const uint64_t&>(rhs));
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
