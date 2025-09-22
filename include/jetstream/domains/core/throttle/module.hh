#ifndef JETSTREAM_DOMAINS_CORE_THROTTLE_MODULE_HH
#define JETSTREAM_DOMAINS_CORE_THROTTLE_MODULE_HH

#include "jetstream/module.hh"

namespace Jetstream::Modules {

struct Throttle : public Module::Config {
    U64 intervalMs = 100;

    JST_MODULE_TYPE(throttle);
    JST_MODULE_PARAMS(intervalMs);
};

}  // namespace Jetstream::Modules

#endif  // JETSTREAM_DOMAINS_CORE_THROTTLE_MODULE_HH
