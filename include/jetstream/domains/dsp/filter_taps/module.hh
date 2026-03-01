#ifndef JETSTREAM_DOMAINS_DSP_FILTER_TAPS_MODULE_HH
#define JETSTREAM_DOMAINS_DSP_FILTER_TAPS_MODULE_HH

#include <vector>

#include "jetstream/module.hh"

namespace Jetstream::Modules {

struct FilterTaps : public Module::Config {
    F64 sampleRate = 2.0e6;
    F64 bandwidth = 1.0e6;
    std::vector<F64> center = {0.0e6};
    U64 taps = 101;

    JST_MODULE_TYPE(filter_taps);
    JST_MODULE_PARAMS(sampleRate, bandwidth, center, taps);
};

}  // namespace Jetstream::Modules

#endif  // JETSTREAM_DOMAINS_DSP_FILTER_TAPS_MODULE_HH
