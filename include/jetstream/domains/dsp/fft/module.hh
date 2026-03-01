#ifndef JETSTREAM_DOMAINS_DSP_FFT_MODULE_HH
#define JETSTREAM_DOMAINS_DSP_FFT_MODULE_HH

#include "jetstream/module.hh"

namespace Jetstream::Modules {

struct Fft : public Module::Config {
    bool forward = true;

    JST_MODULE_TYPE(fft);
    JST_MODULE_PARAMS(forward);
};

}  // namespace Jetstream::Modules

#endif  // JETSTREAM_DOMAINS_DSP_FFT_MODULE_HH
