#ifndef JETSTREAM_DOMAINS_CORE_CAST_MODULE_HH
#define JETSTREAM_DOMAINS_CORE_CAST_MODULE_HH

#include <string>

#include "jetstream/module.hh"
#include "jetstream/types.hh"

namespace Jetstream::Modules {

struct Cast : public Module::Config {
    std::string outputType = "CF32";
    F32 scaler = 0.0f;

    JST_MODULE_TYPE(cast);
    JST_MODULE_PARAMS(outputType, scaler);
};

}  // namespace Jetstream::Modules

#endif  // JETSTREAM_DOMAINS_CORE_CAST_MODULE_HH
