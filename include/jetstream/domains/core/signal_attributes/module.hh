#ifndef JETSTREAM_DOMAINS_CORE_SIGNAL_ATTRIBUTES_MODULE_HH
#define JETSTREAM_DOMAINS_CORE_SIGNAL_ATTRIBUTES_MODULE_HH

#include "jetstream/module.hh"
#include "jetstream/types.hh"

namespace Jetstream::Modules {

struct SignalAttributes : public Module::Config {
    F32 sampleRate = 0.0f;
    F32 frequency = 0.0f;

    JST_MODULE_TYPE(signal_attributes);
    JST_MODULE_PARAMS(sampleRate, frequency);
};

}  // namespace Jetstream::Modules

#endif  // JETSTREAM_DOMAINS_CORE_SIGNAL_ATTRIBUTES_MODULE_HH
