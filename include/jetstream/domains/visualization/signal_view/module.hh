#ifndef JETSTREAM_DOMAINS_VISUALIZATION_SIGNAL_VIEW_MODULE_HH
#define JETSTREAM_DOMAINS_VISUALIZATION_SIGNAL_VIEW_MODULE_HH

#include <string>

#include "jetstream/memory/types.hh"
#include "jetstream/module.hh"

namespace Jetstream::Modules {

struct SignalView : public Module::Config {
    std::string mode = "lineplot";
    U64 averaging = 1;
    U64 decimation = 1;
    bool maxHold = false;
    bool fill = true;
    F32 rangeMin = -100.0f;
    F32 rangeMax = 0.0f;
    U64 waterfallHeight = 1024;
    std::string xLabel = "Frequency (MHz)";
    std::string amplitudeLabel = "Amplitude (dBFS)";
    std::string waterfallLabel = "Time";

    JST_MODULE_TYPE(signal_view);
    JST_MODULE_PARAMS(mode, averaging, decimation, maxHold, fill, rangeMin,
                      rangeMax, waterfallHeight, xLabel,
                      amplitudeLabel, waterfallLabel);
};

}  // namespace Jetstream::Modules

#endif  // JETSTREAM_DOMAINS_VISUALIZATION_SIGNAL_VIEW_MODULE_HH
