#ifndef JETSTREAM_DOMAINS_VISUALIZATION_WATERFALL_MODULE_HH
#define JETSTREAM_DOMAINS_VISUALIZATION_WATERFALL_MODULE_HH

#include "jetstream/memory/types.hh"
#include "jetstream/module.hh"

namespace Jetstream::Modules {

struct Waterfall : public Module::Config {
    U64 height = 512;
    bool interpolate = true;

    JST_MODULE_TYPE(waterfall);
    JST_MODULE_PARAMS(height, interpolate);
};

}  // namespace Jetstream::Modules

#endif  // JETSTREAM_DOMAINS_VISUALIZATION_WATERFALL_MODULE_HH
