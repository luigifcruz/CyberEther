#ifndef JETSTREAM_DOMAINS_DSP_ADSB_MODULE_HH
#define JETSTREAM_DOMAINS_DSP_ADSB_MODULE_HH

#include "jetstream/module.hh"

namespace Jetstream::Modules {

struct Adsb : public Module::Config {
    JST_MODULE_TYPE(adsb);
    JST_MODULE_PARAMS();
};

}  // namespace Jetstream::Modules

#endif  // JETSTREAM_DOMAINS_DSP_ADSB_MODULE_HH
