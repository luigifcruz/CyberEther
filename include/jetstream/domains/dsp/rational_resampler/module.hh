#ifndef JETSTREAM_DOMAINS_DSP_RATIONAL_RESAMPLER_MODULE_HH
#define JETSTREAM_DOMAINS_DSP_RATIONAL_RESAMPLER_MODULE_HH

#include "jetstream/module.hh"

namespace Jetstream::Modules {

struct RationalResampler : public Module::Config {
    U64 interpolation = 1;
    U64 decimation = 1;
    U64 taps = 0;
    F32 cutoff = 0.9f;

    JST_MODULE_TYPE(rational_resampler);
    JST_MODULE_PARAMS(interpolation, decimation, taps, cutoff);
};

}  // namespace Jetstream::Modules

#endif  // JETSTREAM_DOMAINS_DSP_RATIONAL_RESAMPLER_MODULE_HH
