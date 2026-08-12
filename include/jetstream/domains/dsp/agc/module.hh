#ifndef JETSTREAM_DOMAINS_DSP_AGC_MODULE_HH
#define JETSTREAM_DOMAINS_DSP_AGC_MODULE_HH

#include "jetstream/module.hh"

namespace Jetstream::Modules {

struct Agc : public Module::Config {
    U64 tileSize = 1024;
    F64 reference = 1.0;
    F64 epsilon = 1e-12;
    F64 minGain = 0.01;
    F64 maxGain = 100.0;
    F64 maxGainChange = 4.0;

    JST_MODULE_TYPE(agc);
    JST_MODULE_PARAMS(tileSize, reference, epsilon, minGain, maxGain,
                      maxGainChange);
};

}  // namespace Jetstream::Modules

#endif  // JETSTREAM_DOMAINS_DSP_AGC_MODULE_HH
