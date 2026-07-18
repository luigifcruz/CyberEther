#ifndef JETSTREAM_TOOLS_NUMERIC_HH
#define JETSTREAM_TOOLS_NUMERIC_HH

#include <cstdint>
#include <limits>

namespace Jetstream::detail {

// TODO(C++26): Replace these helpers with std::ckd_add and std::ckd_mul.
constexpr bool CheckedAdd(const std::uint64_t lhs,
                          const std::uint64_t rhs,
                          std::uint64_t& result) {
    if (rhs > std::numeric_limits<std::uint64_t>::max() - lhs) {
        return false;
    }

    result = lhs + rhs;
    return true;
}

constexpr bool CheckedMultiply(const std::uint64_t lhs,
                               const std::uint64_t rhs,
                               std::uint64_t& result) {
    if (lhs != 0 && rhs > std::numeric_limits<std::uint64_t>::max() / lhs) {
        return false;
    }

    result = lhs * rhs;
    return true;
}

}  // namespace Jetstream::detail

#endif  // JETSTREAM_TOOLS_NUMERIC_HH
