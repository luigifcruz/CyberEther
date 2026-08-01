#ifndef JETSTREAM_DOMAINS_DSP_OVERLAP_ADD_MODULE_HH
#define JETSTREAM_DOMAINS_DSP_OVERLAP_ADD_MODULE_HH

#include "jetstream/module.hh"

namespace Jetstream::Modules {

struct OverlapAdd : public Module::Config {
    JST_MODULE_TYPE(overlap_add);
    JST_MODULE_PARAMS();
};

}  // namespace Jetstream::Modules

#endif  // JETSTREAM_DOMAINS_DSP_OVERLAP_ADD_MODULE_HH
