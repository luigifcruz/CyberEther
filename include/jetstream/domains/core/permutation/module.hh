#ifndef JETSTREAM_DOMAINS_CORE_PERMUTATION_MODULE_HH
#define JETSTREAM_DOMAINS_CORE_PERMUTATION_MODULE_HH

#include <vector>

#include "jetstream/module.hh"

namespace Jetstream::Modules {

struct Permutation : public Module::Config {
    std::vector<U64> permutation = {0};

    JST_MODULE_TYPE(permutation);
    JST_MODULE_PARAMS(permutation);
};

}  // namespace Jetstream::Modules

#endif  // JETSTREAM_DOMAINS_CORE_PERMUTATION_MODULE_HH
