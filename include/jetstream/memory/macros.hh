#ifndef JETSTREAM_MEMORY_MACROS_HH
#define JETSTREAM_MEMORY_MACROS_HH

#include <unistd.h>
#include <stdlib.h>
#include <stdint.h>

#include "jetstream/macros.hh"

#ifndef JST_PAGESIZE
#ifdef JST_OS_WINDOWS
// I guess this is valid? I don't care about alignment on Windows.
#define JST_PAGESIZE() 4096
#else
#define JST_PAGESIZE() getpagesize()
#endif
#endif 

#ifndef JST_PAGE_ALIGNED_SIZE
#define JST_PAGE_ALIGNED_SIZE(X) (X + JST_PAGESIZE() - 1) & ~(JST_PAGESIZE() - 1)
#endif 

#ifndef JST_IS_ALIGNED
#define JST_IS_ALIGNED(X) (((uintptr_t)(const void *)(X)) % JST_PAGESIZE() == 0)
#endif

#ifndef JST_MIN
#define JST_MIN(a,b) (((a)<(b))?(a):(b))
#endif

#ifndef JST_MHZ
#define JST_MHZ (1000*1000)
#endif

#ifndef JST_MB
#define JST_MB (1024*1024)
#endif

#endif
