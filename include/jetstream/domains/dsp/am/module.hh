#ifndef JETSTREAM_DOMAINS_DSP_AM_MODULE_HH
#define JETSTREAM_DOMAINS_DSP_AM_MODULE_HH

#include "jetstream/module.hh"

namespace Jetstream::Modules {

struct AM : public Module::Config {
    F32 sampleRate = 240e3f;
    F32 dcAlpha = 0.995f;

    JST_MODULE_TYPE(am);
    JST_MODULE_PARAMS(sampleRate, dcAlpha);
};

}  // namespace Jetstream::Modules

#endif  // JETSTREAM_DOMAINS_DSP_AM_MODULE_HH
