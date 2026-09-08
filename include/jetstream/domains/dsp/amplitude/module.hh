#ifndef JETSTREAM_DOMAINS_DSP_AMPLITUDE_MODULE_HH
#define JETSTREAM_DOMAINS_DSP_AMPLITUDE_MODULE_HH

#include "jetstream/module.hh"

namespace Jetstream::Modules {

struct Amplitude : public Module::Config {
    JST_MODULE_TYPE(amplitude);
    JST_MODULE_PARAMS();
};

}  // namespace Jetstream::Modules

#endif  // JETSTREAM_DOMAINS_DSP_AMPLITUDE_MODULE_HH
