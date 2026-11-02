#ifndef JETSTREAM_DOMAINS_VISUALIZATION_EYE_DIAGRAM_MODULE_HH
#define JETSTREAM_DOMAINS_VISUALIZATION_EYE_DIAGRAM_MODULE_HH

#include <string>

#include "jetstream/memory/types.hh"
#include "jetstream/module.hh"

namespace Jetstream::Modules {

struct EyeDiagram : public Module::Config {
    U64 samplesPerSymbol = 8;
    U64 interpolation = 4;
    std::string component = "real";
    F32 rangeMin = -2.0f;
    F32 rangeMax = 2.0f;

    JST_MODULE_TYPE(eye_diagram);
    JST_MODULE_PARAMS(samplesPerSymbol, interpolation, component, rangeMin, rangeMax);
};

}  // namespace Jetstream::Modules

#endif  // JETSTREAM_DOMAINS_VISUALIZATION_EYE_DIAGRAM_MODULE_HH
