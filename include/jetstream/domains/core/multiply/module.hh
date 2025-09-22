#ifndef JETSTREAM_DOMAINS_CORE_MULTIPLY_MODULE_HH
#define JETSTREAM_DOMAINS_CORE_MULTIPLY_MODULE_HH

#include "jetstream/module.hh"

namespace Jetstream::Modules {

struct Multiply : public Module::Config {
    JST_MODULE_TYPE(multiply);
    JST_MODULE_PARAMS();
};

}  // namespace Jetstream::Modules

#endif  // JETSTREAM_DOMAINS_CORE_MULTIPLY_MODULE_HH
