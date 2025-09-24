#ifndef JETSTREAM_MEMORY_MACROS_HH
#define JETSTREAM_MEMORY_MACROS_HH

#include <unistd.h>
#include <stdlib.h>
#include <stdint.h>

#include "jetstream/macros.hh"

#ifndef JST_PAGESIZE
#ifdef JST_OS_WINDOWS
// TODO: Implement JST_PAGESIZE() for Windows.
#define JST_PAGESIZE() 4096
#else
#define JST_PAGESIZE() getpagesize()
#endif
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
