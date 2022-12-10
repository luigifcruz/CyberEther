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

#endif