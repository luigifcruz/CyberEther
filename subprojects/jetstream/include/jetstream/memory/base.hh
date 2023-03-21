#ifndef JETSTREAM_MEMORY_BASE_HH
#define JETSTREAM_MEMORY_BASE_HH

#include "jetstream/types.hh"
#include "jetstream/memory/vector.hh"
#include "jetstream/memory/buffer.hh"
#include "jetstream/memory/macros.hh"

#ifdef JETSTREAM_BACKEND_CPU_AVAILABLE
#include "jetstream/memory/devices/cpu/copy.hh"
#include "jetstream/memory/devices/cpu/vector.hh"
#endif

#ifdef JETSTREAM_BACKEND_METAL_AVAILABLE
#include "jetstream/memory/devices/metal/copy.hh"
#include "jetstream/memory/devices/metal/vector.hh"
#endif

#endif
