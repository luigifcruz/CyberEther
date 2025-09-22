#ifndef JETSTREAM_DOMAINS_DSP_AGC_MODULE_HH
#define JETSTREAM_DOMAINS_DSP_AGC_MODULE_HH

#include "jetstream/module.hh"

namespace Jetstream::Modules {

struct Agc : public Module::Config {
    JST_MODULE_TYPE(agc);
    JST_MODULE_PARAMS();
};

}  // namespace Jetstream::Modules

#endif  // JETSTREAM_DOMAINS_DSP_AGC_MODULE_HH
