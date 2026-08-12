#ifndef JETSTREAM_DOMAINS_CORE_FLATTEN_MODULE_HH
#define JETSTREAM_DOMAINS_CORE_FLATTEN_MODULE_HH

#include "jetstream/module.hh"

namespace Jetstream::Modules {

struct Flatten : public Module::Config {
    JST_MODULE_TYPE(flatten);
    JST_MODULE_PARAMS();
};

}  // namespace Jetstream::Modules

#endif  // JETSTREAM_DOMAINS_CORE_FLATTEN_MODULE_HH
