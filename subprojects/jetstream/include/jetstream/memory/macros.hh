#ifndef JETSTREAM_MEMORY_MACROS_HH
#define JETSTREAM_MEMORY_MACROS_HH

#include <unistd.h>
#include <stdlib.h>

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

#endif
