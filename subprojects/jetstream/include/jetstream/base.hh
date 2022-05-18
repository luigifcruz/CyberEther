#ifndef JETSTREAM_BASE_HH
#define JETSTREAM_BASE_HH

#include "jetstream/types.hh"
#include "jetstream/macros.hh"
#include "jetstream/logger.hh"
#include "jetstream/instance.hh"
#include "jetstream/module.hh"

#include "jetstream/backend/base.hh"
#include "jetstream/render/base.hh"

#ifdef JETSTREAM_MODULE_FFT_CPU_AVAILABLE
#include "jetstream/modules/fft.hh"
#endif

#ifdef JETSTREAM_MODULE_WINDOW_CPU_AVAILABLE
#include "jetstream/modules/window.hh"
#endif

#ifdef JETSTREAM_MODULE_MULTIPLY_CPU_AVAILABLE
#include "jetstream/modules/multiply.hh"
#endif

#ifdef JETSTREAM_MODULE_AMPLITUDE_CPU_AVAILABLE
#include "jetstream/modules/amplitude.hh"
#endif

#ifdef JETSTREAM_MODULE_SCALE_CPU_AVAILABLE
#include "jetstream/modules/scale.hh"
#endif

#endif
