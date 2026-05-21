#ifndef BLUEPRINT_GAIN_MODULE_HH
#define BLUEPRINT_GAIN_MODULE_HH

#include <jetstream/module.hh>
#include <jetstream/types.hh>

namespace Jetstream::Modules {

struct BlueprintGain : public Module::Config {
    F32 gain = 1.0f;

    JST_MODULE_TYPE(blueprint_gain);
    JST_MODULE_PARAMS(gain);
};

}  // namespace Jetstream::Modules

#endif  // BLUEPRINT_GAIN_MODULE_HH
