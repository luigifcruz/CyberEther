#ifndef JETSTREAM_DOMAINS_CORE_COMPLEX_SPLIT_MODULE_HH
#define JETSTREAM_DOMAINS_CORE_COMPLEX_SPLIT_MODULE_HH

#include <string>

#include "jetstream/module.hh"
#include "jetstream/types.hh"

namespace Jetstream::Modules {

struct ComplexSplit : public Module::Config {
    std::string mode = "rectangular";

    JST_MODULE_TYPE(complex_split);
    JST_MODULE_PARAMS(mode);
};

}  // namespace Jetstream::Modules

#endif  // JETSTREAM_DOMAINS_CORE_COMPLEX_SPLIT_MODULE_HH
