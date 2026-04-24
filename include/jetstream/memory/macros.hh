#ifndef JETSTREAM_MEMORY_MACROS_HH
#define JETSTREAM_MEMORY_MACROS_HH

#include <cstddef>
#include <cstdint>
#include <cstdlib>

#include "jetstream/config.hh"

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

}  // namespace Jetstream::detail

#ifndef JST_PAGESIZE
#define JST_PAGESIZE() ::Jetstream::detail::SystemPageSize()
#endif

#ifndef JST_ROUND_UP
#define JST_ROUND_UP(X, Y) (((X) + (Y) - 1) / (Y)) * (Y)
#endif

#ifndef JST_PAGE_ALIGNED_SIZE
#define JST_PAGE_ALIGNED_SIZE(X) JST_ROUND_UP(X, JST_PAGESIZE())
#endif

#ifndef JST_IS_ALIGNED
#define JST_IS_ALIGNED(X) (((uintptr_t)(const void *)(X)) % JST_PAGESIZE() == 0)
#endif

#endif  // JETSTREAM_MEMORY_MACROS_HH
