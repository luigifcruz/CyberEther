#ifndef JETSTREAM_LPT_BASE_H
#define JETSTREAM_LPT_BASE_H

#include "jetstream/modules/lineplot/generic.hpp"
#include "jetstream/modules/lineplot/cpu.hpp"
#ifdef JETSTREAM_LPT_CUDA_AVAILABLE
#include "jetstream/modules/lineplot/cuda.hpp"
#endif

#endif
