#ifndef JETSTREAM_MEMORY_MACROS_HH
#define JETSTREAM_MEMORY_MACROS_HH

#include <cstddef>
#include <cstdint>
#include <cstdlib>

#include "jetstream/config.hh"
#include "jetstream/tools/numeric.hh"

#if defined(JST_OS_WINDOWS)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#undef ERROR
#undef FATAL
#else
#include <unistd.h>
#endif

namespace Jetstream::detail {

inline std::size_t SystemPageSize() {
#if defined(JST_OS_WINDOWS)
    SYSTEM_INFO info{};
    GetSystemInfo(&info);
    return static_cast<std::size_t>(info.dwPageSize);
#else
    return static_cast<std::size_t>(getpagesize());
#endif
}

inline bool CheckedRoundUp(const std::uint64_t value, const std::uint64_t alignment, std::uint64_t& rounded) {
    if (alignment == 0) {
        return false;
    }

    const auto remainder = value % alignment;
    const auto padding = remainder == 0 ? 0 : alignment - remainder;
    return CheckedAdd(value, padding, rounded);
}

inline bool CheckedPageAlignedSize(const std::uint64_t bytes, std::uint64_t& alignedBytes) {
    return CheckedRoundUp(bytes, SystemPageSize(), alignedBytes);
}

}  // namespace Jetstream::detail

#ifndef JST_PAGESIZE
#define JST_PAGESIZE() ::Jetstream::detail::SystemPageSize()
#endif

#ifndef JST_IS_ALIGNED
#define JST_IS_ALIGNED(X) (((uintptr_t)(const void *)(X)) % JST_PAGESIZE() == 0)
#endif

#endif  // JETSTREAM_MEMORY_MACROS_HH
