#ifndef JETSTREAM_DOMAINS_CORE_RANGE_MODULE_HH
#define JETSTREAM_DOMAINS_CORE_RANGE_MODULE_HH

#include "jetstream/module.hh"
#include "jetstream/types.hh"

namespace Jetstream::Modules {

struct Range : public Module::Config {
    F32 min = -1.0f;
    F32 max = +1.0f;

    JST_MODULE_TYPE(range);
    JST_MODULE_PARAMS(min, max);
};

}  // namespace Jetstream::Modules

#endif  // JETSTREAM_DOMAINS_CORE_RANGE_MODULE_HH
