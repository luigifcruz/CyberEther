#ifndef JETSTREAM_DOMAINS_CORE_INVERT_MODULE_HH
#define JETSTREAM_DOMAINS_CORE_INVERT_MODULE_HH

#include "jetstream/module.hh"

namespace Jetstream::Modules {

struct Invert : public Module::Config {
    JST_MODULE_TYPE(invert);
    JST_MODULE_PARAMS();
};

}  // namespace Jetstream::Modules

#endif  // JETSTREAM_DOMAINS_CORE_INVERT_MODULE_HH
