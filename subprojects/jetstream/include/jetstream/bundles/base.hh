#ifndef JETSTREAM_BUNDLES_BASE_HH
#define JETSTREAM_BUNDLES_BASE_HH

#include "jetstream/types.hh"

#if defined(JETSTREAM_MODULE_LINEPLOT_CPU_AVAILABLE) || \
    defined(JETSTREAM_MODULE_LINEPLOT_METAL_AVAILABLE)
#include "jetstream/bundles/lineplot.hh"
#endif

#if defined(JETSTREAM_MODULE_WATERFALL_CPU_AVAILABLE) || \
    defined(JETSTREAM_MODULE_WATERFALL_METAL_AVAILABLE)
#include "jetstream/bundles/waterfall.hh"
#endif

#if defined(JETSTREAM_MODULE_SPECTROGRAM_CPU_AVAILABLE) || \
    defined(JETSTREAM_MODULE_SPECTROGRAM_METAL_AVAILABLE)
#include "jetstream/bundles/spectrogram.hh"
#endif

#if defined(JETSTREAM_MODULE_CONSTELLATION_CPU_AVAILABLE) || \
    defined(JETSTREAM_MODULE_CONSTELLATION_METAL_AVAILABLE)
#include "jetstream/bundles/constellation.hh"
#endif

#if defined(JETSTREAM_MODULE_CONSTELLATION_CPU_AVAILABLE) || \
    defined(JETSTREAM_MODULE_CONSTELLATION_METAL_AVAILABLE)
#include "jetstream/bundles/constellation.hh"
#endif

#if defined(JETSTREAM_MODULE_SOAPY_CPU_AVAILABLE) || \
    defined(JETSTREAM_MODULE_SOAPY_METAL_AVAILABLE)
#include "jetstream/bundles/soapy.hh"
#endif


#endif
