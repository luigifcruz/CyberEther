#ifndef JETSTREAM_DOMAINS_IO_SDR_MODULE_HH
#define JETSTREAM_DOMAINS_IO_SDR_MODULE_HH

#include <vector>

#include "jetstream/memory/types.hh"
#include "jetstream/module.hh"

namespace Jetstream::Modules {

struct Sdr : public Module::Config {
    std::string deviceString = "";
    std::string antenna = "";
    U64 channels = 1;
    F32 frequency = 96.9e6;
    F32 sampleRate = 2.0e6;
    bool automaticGain = true;
    F32 manualGain = 30.0f;
    std::vector<F32> manualGains;
    bool biasTee = false;
    U64 numberOfBatches = 8;
    U64 numberOfTimeSamples = 8192;
    U64 bufferMultiplier = 4;

    JST_MODULE_TYPE(sdr);
    JST_MODULE_PARAMS(deviceString, antenna, channels, frequency, sampleRate,
                      automaticGain, manualGain, manualGains, biasTee, numberOfBatches,
                      numberOfTimeSamples, bufferMultiplier);
};

}  // namespace Jetstream::Modules

#endif  // JETSTREAM_DOMAINS_IO_SDR_MODULE_HH
