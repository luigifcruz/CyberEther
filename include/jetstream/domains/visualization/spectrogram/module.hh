#ifndef JETSTREAM_DOMAINS_VISUALIZATION_SPECTROGRAM_MODULE_HH
#define JETSTREAM_DOMAINS_VISUALIZATION_SPECTROGRAM_MODULE_HH

#include "jetstream/memory/types.hh"
#include "jetstream/module.hh"

namespace Jetstream::Modules {

struct Spectrogram : public Module::Config {
    U64 height = 256;

    JST_MODULE_TYPE(spectrogram);
    JST_MODULE_PARAMS(height);
};

}  // namespace Jetstream::Modules

#endif  // JETSTREAM_DOMAINS_VISUALIZATION_SPECTROGRAM_MODULE_HH
