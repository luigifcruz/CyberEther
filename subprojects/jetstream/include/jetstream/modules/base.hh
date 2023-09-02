#ifndef JETSTREAM_MODULES_BASE_HH
#define JETSTREAM_MODULES_BASE_HH

#include "jetstream/types.hh"

// 
// Compute
//

#if defined(JETSTREAM_MODULE_FFT_CPU_AVAILABLE) || \
    defined(JETSTREAM_MODULE_FFT_METAL_AVAILABLE)
#include "jetstream/modules/fft.hh"
#endif

#if defined(JETSTREAM_MODULE_FILTER_CPU_AVAILABLE) || \
    defined(JETSTREAM_MODULE_FILTER_METAL_AVAILABLE)
#include "jetstream/modules/filter.hh"
#endif

#if defined(JETSTREAM_MODULE_WINDOW_CPU_AVAILABLE) || \
    defined(JETSTREAM_MODULE_WINDOW_METAL_AVAILABLE)
#include "jetstream/modules/window.hh"
#endif

#if defined(JETSTREAM_MODULE_MULTIPLY_CPU_AVAILABLE) || \
    defined(JETSTREAM_MODULE_MULTIPLY_METAL_AVAILABLE)
#include "jetstream/modules/multiply.hh"
#endif

#if defined(JETSTREAM_MODULE_AMPLITUDE_CPU_AVAILABLE) || \
    defined(JETSTREAM_MODULE_AMPLITUDE_METAL_AVAILABLE)
#include "jetstream/modules/amplitude.hh"
#endif

#if defined(JETSTREAM_MODULE_SCALE_CPU_AVAILABLE) || \
    defined(JETSTREAM_MODULE_SCALE_METAL_AVAILABLE)
#include "jetstream/modules/scale.hh"
#endif

#if defined(JETSTREAM_MODULE_SOAPY_CPU_AVAILABLE) || \
    defined(JETSTREAM_MODULE_SOAPY_METAL_AVAILABLE)
#include "jetstream/modules/soapy.hh"
#endif

#if defined(JETSTREAM_MODULE_AUDIO_CPU_AVAILABLE) || \
    defined(JETSTREAM_MODULE_AUDIO_METAL_AVAILABLE)
#include "jetstream/modules/audio.hh"
#endif

#if defined(JETSTREAM_MODULE_FM_CPU_AVAILABLE) || \
    defined(JETSTREAM_MODULE_FM_METAL_AVAILABLE)
#include "jetstream/modules/fm.hh"
#endif

#if defined(JETSTREAM_MODULE_MULTIPLY_CONSTANT_CPU_AVAILABLE) || \
    defined(JETSTREAM_MODULE_MULTIPLY_CONSTANT_METAL_AVAILABLE)
#include "jetstream/modules/multiply_constant.hh"
#endif

//
// Graphical
// 

#if defined(JETSTREAM_MODULE_LINEPLOT_CPU_AVAILABLE) || \
    defined(JETSTREAM_MODULE_LINEPLOT_METAL_AVAILABLE)
#include "jetstream/modules/lineplot.hh"
#endif

#if defined(JETSTREAM_MODULE_WATERFALL_CPU_AVAILABLE) || \
    defined(JETSTREAM_MODULE_WATERFALL_METAL_AVAILABLE)
#include "jetstream/modules/waterfall.hh"
#endif

#if defined(JETSTREAM_MODULE_SPECTROGRAM_CPU_AVAILABLE) || \
    defined(JETSTREAM_MODULE_SPECTROGRAM_METAL_AVAILABLE)
#include "jetstream/modules/spectrogram.hh"
#endif

#if defined(JETSTREAM_MODULE_CONSTELLATION_CPU_AVAILABLE) || \
    defined(JETSTREAM_MODULE_CONSTELLATION_METAL_AVAILABLE)
#include "jetstream/modules/constellation.hh"
#endif

#endif
