#ifndef JETSTREAM_DOMAINS_DSP_SQUELCH_MODULE_HH
#define JETSTREAM_DOMAINS_DSP_SQUELCH_MODULE_HH

#include "jetstream/module.hh"

namespace Jetstream::Modules {

struct Squelch : public Module::Config {
    F32 threshold = 0.1f;

    JST_MODULE_TYPE(squelch);
    JST_MODULE_PARAMS(threshold);
};

}  // namespace Jetstream::Modules

#endif  // JETSTREAM_DOMAINS_DSP_SQUELCH_MODULE_HH
