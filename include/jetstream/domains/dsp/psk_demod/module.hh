#ifndef JETSTREAM_DOMAINS_DSP_PSK_DEMOD_MODULE_HH
#define JETSTREAM_DOMAINS_DSP_PSK_DEMOD_MODULE_HH

#include "jetstream/module.hh"

namespace Jetstream::Modules {

struct PskDemod : public Module::Config {
    std::string pskType = "qpsk";
    F64 sampleRate = 2000000.0;
    F64 symbolRate = 1000000.0;
    F64 frequencyLoopBandwidth = 0.05;
    F64 timingLoopBandwidth = 0.05;
    F64 dampingFactor = 0.707;

    JST_MODULE_TYPE(psk_demod);
    JST_MODULE_PARAMS(pskType, sampleRate, symbolRate, frequencyLoopBandwidth,
                      timingLoopBandwidth, dampingFactor);
};

}  // namespace Jetstream::Modules

#endif  // JETSTREAM_DOMAINS_DSP_PSK_DEMOD_MODULE_HH
