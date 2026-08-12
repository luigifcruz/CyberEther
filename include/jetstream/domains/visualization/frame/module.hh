#ifndef JETSTREAM_DOMAINS_VISUALIZATION_FRAME_MODULE_HH
#define JETSTREAM_DOMAINS_VISUALIZATION_FRAME_MODULE_HH

#include "jetstream/memory/types.hh"
#include "jetstream/module.hh"

namespace Jetstream::Modules {

struct Frame : public Module::Config {
    bool lut = false;

    JST_MODULE_TYPE(frame);
    JST_MODULE_PARAMS(lut);
};

}  // namespace Jetstream::Modules

#endif  // JETSTREAM_DOMAINS_VISUALIZATION_FRAME_MODULE_HH
