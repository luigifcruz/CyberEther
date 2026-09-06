#ifndef JETSTREAM_DOMAINS_VISUALIZATION_SPECTROGRAM_MODULE_HH
#define JETSTREAM_DOMAINS_VISUALIZATION_SPECTROGRAM_MODULE_HH

#include <string>

#include "jetstream/memory/types.hh"
#include "jetstream/module.hh"

namespace Jetstream::Modules {

struct Spectrogram : public Module::Config {
    U64 height = 256;
    std::string xLabel = "Frequency (MHz)";
    std::string yLabel = "Magnitude";

    JST_MODULE_TYPE(spectrogram);
    JST_MODULE_PARAMS(height, xLabel, yLabel);
};

}  // namespace Jetstream::Modules

#endif  // JETSTREAM_DOMAINS_VISUALIZATION_SPECTROGRAM_MODULE_HH
