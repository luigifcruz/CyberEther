#ifndef JETSTREAM_MEMORY_AXIS_HH
#define JETSTREAM_MEMORY_AXIS_HH

#include <limits>
#include <optional>

#include "jetstream/memory/types.hh"

namespace Jetstream {

inline std::optional<Index> ResolveAxis(const I64 axis, const Index rank) {
    if (rank == 0 || rank > static_cast<Index>(std::numeric_limits<I64>::max())) {
        return std::nullopt;
    }

    const I64 signedRank = static_cast<I64>(rank);
    const I64 resolvedAxis = axis < 0 ? signedRank + axis : axis;
    if (resolvedAxis < 0 || resolvedAxis >= signedRank) {
        return std::nullopt;
    }

    return static_cast<Index>(resolvedAxis);
}

inline std::optional<Index> ResolveInsertionAxis(const I64 axis, const Index rank) {
    if (rank > static_cast<Index>(std::numeric_limits<I64>::max())) {
        return std::nullopt;
    }

    const I64 signedRank = static_cast<I64>(rank);
    if (axis >= 0) {
        if (axis > signedRank) {
            return std::nullopt;
        }
        return static_cast<Index>(axis);
    }

    const I64 resolvedAxisMinusOne = signedRank + axis;
    if (resolvedAxisMinusOne < -1) {
        return std::nullopt;
    }

    return static_cast<Index>(resolvedAxisMinusOne + 1);
}

}  // namespace Jetstream

#endif  // JETSTREAM_MEMORY_AXIS_HH
