#ifndef JETSTREAM_DOMAINS_DSP_FFT_MODULE_HH
#define JETSTREAM_DOMAINS_DSP_FFT_MODULE_HH

#include "jetstream/module.hh"

namespace Jetstream::Modules {

struct Fft : public Module::Config {
    bool forward = true;
    bool invert = false;
    bool complexOutput = false;

    JST_MODULE_TYPE(fft);
    JST_MODULE_PARAMS(forward, invert, complexOutput);
};

}  // namespace Jetstream::Modules

#endif  // JETSTREAM_DOMAINS_DSP_FFT_MODULE_HH
