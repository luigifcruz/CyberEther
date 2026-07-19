#ifndef JETSTREAM_DOMAINS_DSP_OVERLAP_ADD_MODULE_HH
#define JETSTREAM_DOMAINS_DSP_OVERLAP_ADD_MODULE_HH

#include "jetstream/module.hh"

namespace Jetstream::Modules {

struct OverlapAdd : public Module::Config {
    I64 axis = -1;

    JST_MODULE_TYPE(overlap_add);
    JST_MODULE_PARAMS(axis);
};

}  // namespace Jetstream::Modules

#endif  // JETSTREAM_DOMAINS_DSP_OVERLAP_ADD_MODULE_HH
