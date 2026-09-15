#ifndef JETSTREAM_DOMAINS_CORE_REMOVE_INDICES_MODULE_HH
#define JETSTREAM_DOMAINS_CORE_REMOVE_INDICES_MODULE_HH

#include <vector>

#include "jetstream/module.hh"
#include "jetstream/types.hh"

namespace Jetstream::Modules {

struct RemoveIndices : public Module::Config {
    I64 axis = -1;
    std::vector<U64> indices = {};

    JST_MODULE_TYPE(remove_indices);
    JST_MODULE_PARAMS(axis, indices);
};

}  // namespace Jetstream::Modules

#endif  // JETSTREAM_DOMAINS_CORE_REMOVE_INDICES_MODULE_HH
