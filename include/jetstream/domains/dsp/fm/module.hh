#ifndef JETSTREAM_DOMAINS_DSP_FM_MODULE_HH
#define JETSTREAM_DOMAINS_DSP_FM_MODULE_HH

#include "jetstream/module.hh"

namespace Jetstream::Modules {

struct FM : public Module::Config {
    F32 sampleRate = 240e3f;

    JST_MODULE_TYPE(fm);
    JST_MODULE_PARAMS(sampleRate);
};

}  // namespace Jetstream::Modules

#endif  // JETSTREAM_DOMAINS_DSP_FM_MODULE_HH
