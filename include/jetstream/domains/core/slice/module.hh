#ifndef JETSTREAM_DOMAINS_CORE_SLICE_MODULE_HH
#define JETSTREAM_DOMAINS_CORE_SLICE_MODULE_HH

#include "jetstream/module.hh"

namespace Jetstream::Modules {

struct Slice : public Module::Config {
    std::string slice = "[...]";

    JST_MODULE_TYPE(slice);
    JST_MODULE_PARAMS(slice);
};

}  // namespace Jetstream::Modules

#endif  // JETSTREAM_DOMAINS_CORE_SLICE_MODULE_HH
