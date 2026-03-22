#ifndef JETSTREAM_DOMAINS_CORE_ARITHMETIC_MODULE_HH
#define JETSTREAM_DOMAINS_CORE_ARITHMETIC_MODULE_HH

#include <string>

#include "jetstream/module.hh"
#include "jetstream/types.hh"

namespace Jetstream::Modules {

struct Arithmetic : public Module::Config {
    std::string operation = "add";
    U64 axis = 0;
    bool squeeze = false;

    JST_MODULE_TYPE(arithmetic);
    JST_MODULE_PARAMS(operation, axis, squeeze);
};

}  // namespace Jetstream::Modules

#endif  // JETSTREAM_DOMAINS_CORE_ARITHMETIC_MODULE_HH
