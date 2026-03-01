#ifndef JETSTREAM_DOMAINS_VISUALIZATION_LINEPLOT_MODULE_HH
#define JETSTREAM_DOMAINS_VISUALIZATION_LINEPLOT_MODULE_HH

#include "jetstream/memory/types.hh"
#include "jetstream/module.hh"

namespace Jetstream::Modules {

struct Lineplot : public Module::Config {
    U64 averaging = 1;
    U64 decimation = 1;
    U64 numberOfVerticalLines = 11;
    U64 numberOfHorizontalLines = 5;
    F32 thickness = 1.0f;

    JST_MODULE_TYPE(lineplot);
    JST_MODULE_PARAMS(averaging, decimation, numberOfVerticalLines, numberOfHorizontalLines, thickness);
};

}  // namespace Jetstream::Modules

#endif  // JETSTREAM_DOMAINS_VISUALIZATION_LINEPLOT_MODULE_HH
