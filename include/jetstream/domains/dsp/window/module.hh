#ifndef JETSTREAM_DOMAINS_DSP_WINDOW_MODULE_HH
#define JETSTREAM_DOMAINS_DSP_WINDOW_MODULE_HH

#include "jetstream/module.hh"

namespace Jetstream::Modules {

struct Window : public Module::Config {
    U64 size = 1024;

    JST_MODULE_TYPE(window);
    JST_MODULE_PARAMS(size);
};

}  // namespace Jetstream::Modules

#endif  // JETSTREAM_DOMAINS_DSP_WINDOW_MODULE_HH
