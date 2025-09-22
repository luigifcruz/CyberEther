#ifndef JETSTREAM_DOMAINS_IO_SOAPY_MODULE_HH
#define JETSTREAM_DOMAINS_IO_SOAPY_MODULE_HH

#include "jetstream/memory/types.hh"
#include "jetstream/module.hh"

namespace Jetstream::Modules {

struct Soapy : public Module::Config {
    std::string deviceString = "";
    std::string streamString = "";
    F32 frequency = 96.9e6;
    F32 sampleRate = 2.0e6;
    bool automaticGain = true;
    U64 numberOfBatches = 8;
    U64 numberOfTimeSamples = 8192;
    U64 bufferMultiplier = 4;

    JST_MODULE_TYPE(soapy);
    JST_MODULE_PARAMS(deviceString, streamString, frequency, sampleRate,
                      automaticGain, numberOfBatches, numberOfTimeSamples,
                      bufferMultiplier);
};

}  // namespace Jetstream::Modules

#endif  // JETSTREAM_DOMAINS_IO_SOAPY_MODULE_HH
