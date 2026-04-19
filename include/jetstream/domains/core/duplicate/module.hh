#ifndef JETSTREAM_DOMAINS_CORE_DUPLICATE_MODULE_HH
#define JETSTREAM_DOMAINS_CORE_DUPLICATE_MODULE_HH

#include "jetstream/module.hh"

namespace Jetstream::Modules {

struct Duplicate : public Module::Config {
    std::string outputDevice = "cpu";
    bool hostAccessible = true;

    JST_MODULE_TYPE(duplicate);
    JST_MODULE_PARAMS(hostAccessible, outputDevice);
};

}  // namespace Jetstream::Modules

#endif  // JETSTREAM_DOMAINS_CORE_DUPLICATE_MODULE_HH
