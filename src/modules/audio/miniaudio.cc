#include "jetstream/logger.hh"

#ifdef JST_OS_BROWSER
#define MA_EMSCRIPTEN
#define MA_ENABLE_AUDIO_WORKLETS
#endif

#ifdef JST_DEBUG_MODE
#define MA_DEBUG_OUTPUT
#endif

#define MINIAUDIO_IMPLEMENTATION
#include "miniaudio.h"
