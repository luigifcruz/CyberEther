#ifndef JETSTREAM_MEMORY_BASE_HH
#define JETSTREAM_MEMORY_BASE_HH

#include "jetstream/types.hh"
#include "jetstream/memory/vector.hh"
#include "jetstream/memory/buffer.hh"

#include "jetstream/memory/devices/cpu/copy.hh"
#include "jetstream/memory/devices/cpu/vector.hh"

#ifdef JETSTREAM_CUDA_AVAILABLE

#include "jetstream/memory/devices/cuda/copy.hh"
#include "jetstream/memory/devices/cuda/vector.hh"
#include "jetstream/memory/devices/cuda/helper.hh"

#endif

#endif
