#ifndef JETSTREAM_DOMAINS_CORE_ADD_MODULE_HH
#define JETSTREAM_DOMAINS_CORE_ADD_MODULE_HH

#include "jetstream/module.hh"

namespace Jetstream::Modules {

struct Add : public Module::Config {
    JST_MODULE_TYPE(add);
    JST_MODULE_PARAMS();
};

}  // namespace Jetstream::Modules

#endif  // JETSTREAM_DOMAINS_CORE_ADD_MODULE_HH
