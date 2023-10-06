#include "jetstream/logger.hh"

#ifdef JST_DEBUG_MODE
#define MA_DEBUG_OUTPUT 1
#endif

#define MINIAUDIO_IMPLEMENTATION
#include "jetstream/tools/miniaudio.h"