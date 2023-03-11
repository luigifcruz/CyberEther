#ifndef JETSTREAM_MEMORY_BASE_HH
#define JETSTREAM_MEMORY_BASE_HH

#include "jetstream/types.hh"
#include "jetstream/memory/vector.hh"
#include "jetstream/memory/buffer.hh"

#ifdef JETSTREAM_BACKEND_CPU_AVAILABLE
#include "jetstream/memory/devices/cpu/copy.hh"
#include "jetstream/memory/devices/cpu/vector.hh"
#endif

#endif
