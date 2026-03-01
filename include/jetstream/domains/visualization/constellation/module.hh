#ifndef JETSTREAM_DOMAINS_VISUALIZATION_CONSTELLATION_MODULE_HH
#define JETSTREAM_DOMAINS_VISUALIZATION_CONSTELLATION_MODULE_HH

#include "jetstream/memory/types.hh"
#include "jetstream/module.hh"

namespace Jetstream::Modules {

struct Constellation : public Module::Config {
    JST_MODULE_TYPE(constellation);
    JST_MODULE_PARAMS();
};

}  // namespace Jetstream::Modules

#endif  // JETSTREAM_DOMAINS_VISUALIZATION_CONSTELLATION_MODULE_HH
