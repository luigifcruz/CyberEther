#ifndef JETSTREAM_DOMAINS_DSP_RRC_FILTER_MODULE_HH
#define JETSTREAM_DOMAINS_DSP_RRC_FILTER_MODULE_HH

#include "jetstream/module.hh"

namespace Jetstream::Modules {

struct RrcFilter : public Module::Config {
    F32 symbolRate = 1.0e6f;
    F32 sampleRate = 2.0e6f;
    F32 rollOff = 0.35f;
    U64 taps = 101;

    JST_MODULE_TYPE(rrc_filter);
    JST_MODULE_PARAMS(symbolRate, sampleRate, rollOff, taps);
};

}  // namespace Jetstream::Modules

#endif  // JETSTREAM_DOMAINS_DSP_RRC_FILTER_MODULE_HH
