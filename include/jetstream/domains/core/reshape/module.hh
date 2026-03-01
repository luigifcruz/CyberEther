#ifndef JETSTREAM_DOMAINS_CORE_RESHAPE_MODULE_HH
#define JETSTREAM_DOMAINS_CORE_RESHAPE_MODULE_HH

#include "jetstream/module.hh"

namespace Jetstream::Modules {

struct Reshape : public Module::Config {
    std::string shape = "[]";

    JST_MODULE_TYPE(reshape);
    JST_MODULE_PARAMS(shape);
};

}  // namespace Jetstream::Modules

#endif  // JETSTREAM_DOMAINS_CORE_RESHAPE_MODULE_HH
