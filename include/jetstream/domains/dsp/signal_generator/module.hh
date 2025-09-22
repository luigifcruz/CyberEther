#ifndef JETSTREAM_DOMAINS_DSP_SIGNAL_GENERATOR_MODULE_HH
#define JETSTREAM_DOMAINS_DSP_SIGNAL_GENERATOR_MODULE_HH

#include "jetstream/memory/types.hh"
#include "jetstream/module.hh"

namespace Jetstream::Modules {

struct SignalGenerator : public Module::Config {
    std::string signalType = "cosine";
    std::string signalDataType = "F32";
    F64 sampleRate = 1000000.0;
    F64 frequency = 1000.0;
    F64 amplitude = 1.0;
    F64 phase = 0.0;
    F64 dcOffset = 0.0;
    F64 noiseVariance = 1.0;
    F64 chirpStartFreq = 1000.0;
    F64 chirpEndFreq = 10000.0;
    F64 chirpDuration = 1.0;
    U64 bufferSize = 8192;

    JST_MODULE_TYPE(signal_generator);
    JST_MODULE_PARAMS(signalType, signalDataType, sampleRate, frequency, amplitude, phase, dcOffset,
                      noiseVariance, chirpStartFreq, chirpEndFreq, chirpDuration,
                      bufferSize);
};

}  // namespace Jetstream::Modules

#endif  // JETSTREAM_DOMAINS_DSP_SIGNAL_GENERATOR_MODULE_HH
