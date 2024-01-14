#include "jetstream/logger.hh"

#ifdef JST_DEBUG_MODE
#define MA_DEBUG_OUTPUT
#endif

#define MA_NO_RUNTIME_LINKING

#define MINIAUDIO_IMPLEMENTATION
#include "miniaudio.h"