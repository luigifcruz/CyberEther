#ifndef JETSTREAM_DOMAINS_CORE_COMPARATOR_MODULE_HH
#define JETSTREAM_DOMAINS_CORE_COMPARATOR_MODULE_HH

#include "jetstream/module.hh"

namespace Jetstream::Modules {

struct Comparator : public Module::Config {
    U64 inputCount = 2;
    F64 tolerance = 1e-6;

    JST_MODULE_TYPE(comparator);
    JST_MODULE_PARAMS(inputCount, tolerance);
};

}  // namespace Jetstream::Modules

#endif  // JETSTREAM_DOMAINS_CORE_COMPARATOR_MODULE_HH
