#ifndef JETSTREAM_DOMAINS_IO_SDR_SWEEP_MODULE_HH
#define JETSTREAM_DOMAINS_IO_SDR_SWEEP_MODULE_HH

#include "jetstream/memory/types.hh"
#include "jetstream/module.hh"

namespace Jetstream::Modules {

struct SdrSweep : public Module::Config {
    std::string deviceString = "";
    std::string antenna = "";
    bool automaticGain = false;
    F32 manualGain = 30.0f;
    bool biasTee = false;
    F64 startFrequency = 88.0e6;
    F64 endFrequency = 108.0e6;
    F64 sampleRate = 0.0;
    F64 resolution = 10.0e3;
    U64 dwell = 1;
    F32 settlingTime = 0.0f;

    JST_MODULE_TYPE(sdr_sweep);
    JST_MODULE_PARAMS(deviceString, antenna, automaticGain, manualGain, biasTee,
                      startFrequency, endFrequency, sampleRate, resolution,
                      dwell, settlingTime);
};

}  // namespace Jetstream::Modules

#endif  // JETSTREAM_DOMAINS_IO_SDR_SWEEP_MODULE_HH
