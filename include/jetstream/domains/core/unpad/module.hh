#ifndef JETSTREAM_DOMAINS_CORE_UNPAD_MODULE_HH
#define JETSTREAM_DOMAINS_CORE_UNPAD_MODULE_HH

#include "jetstream/module.hh"
#include "jetstream/types.hh"

namespace Jetstream::Modules {

struct Unpad : public Module::Config {
    U64 size = 0;
    U64 axis = 0;

    JST_MODULE_TYPE(unpad);
    JST_MODULE_PARAMS(size, axis);
};

}  // namespace Jetstream::Modules

#endif  // JETSTREAM_DOMAINS_CORE_UNPAD_MODULE_HH
