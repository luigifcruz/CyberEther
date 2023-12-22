#ifndef JETSTREAM_MEMORY_MACROS_HH
#define JETSTREAM_MEMORY_MACROS_HH

#include <unistd.h>
#include <stdlib.h>
#include <stdint.h>

#ifndef JST_PAGESIZE
#define JST_PAGESIZE() getpagesize()
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

inline uint64_t HashU64(uint64_t x) {
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebull;
    x = (x ^ (x >> 31));
    return x;
}

#endif
