#ifndef JETSTREAM_DOMAINS_CORE_SIGNAL_AXES_MODULE_HH
#define JETSTREAM_DOMAINS_CORE_SIGNAL_AXES_MODULE_HH

#include <string>

#include "jetstream/module.hh"

namespace Jetstream::Modules {

struct SignalAxes : public Module::Config {
    std::string axes;

    JST_MODULE_TYPE(signal_axes);
    JST_MODULE_PARAMS(axes);
};

}  // namespace Jetstream::Modules

#endif  // JETSTREAM_DOMAINS_CORE_SIGNAL_AXES_MODULE_HH
