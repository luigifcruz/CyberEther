#ifndef JETSTREAM_DOMAINS_CORE_EXPAND_DIMS_MODULE_HH
#define JETSTREAM_DOMAINS_CORE_EXPAND_DIMS_MODULE_HH

#include "jetstream/module.hh"

namespace Jetstream::Modules {

struct ExpandDims : public Module::Config {
    U64 axis = 0;

    JST_MODULE_TYPE(expand_dims);
    JST_MODULE_PARAMS(axis);
};

}  // namespace Jetstream::Modules

#endif  // JETSTREAM_DOMAINS_CORE_EXPAND_DIMS_MODULE_HH
