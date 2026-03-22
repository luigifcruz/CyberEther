#ifndef JETSTREAM_DOMAINS_CORE_SQUEEZE_DIMS_MODULE_HH
#define JETSTREAM_DOMAINS_CORE_SQUEEZE_DIMS_MODULE_HH

#include "jetstream/module.hh"
#include "jetstream/types.hh"

namespace Jetstream::Modules {

struct SqueezeDims : public Module::Config {
    U64 axis = 0;

    JST_MODULE_TYPE(squeeze_dims);
    JST_MODULE_PARAMS(axis);
};

}  // namespace Jetstream::Modules

#endif  // JETSTREAM_DOMAINS_CORE_SQUEEZE_DIMS_MODULE_HH
