#ifndef JETSTREAM_DOMAINS_DSP_FOLD_MODULE_HH
#define JETSTREAM_DOMAINS_DSP_FOLD_MODULE_HH

#include "jetstream/module.hh"

namespace Jetstream::Modules {

struct Fold : public Module::Config {
    U64 axis = 0;
    U64 offset = 0;
    U64 size = 0;

    JST_MODULE_TYPE(fold);
    JST_MODULE_PARAMS(axis, offset, size);
};

}  // namespace Jetstream::Modules

#endif  // JETSTREAM_DOMAINS_DSP_FOLD_MODULE_HH
