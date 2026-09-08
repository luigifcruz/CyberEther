#ifndef JETSTREAM_DOMAINS_DSP_PHASE_CORRECTION_MODULE_HH
#define JETSTREAM_DOMAINS_DSP_PHASE_CORRECTION_MODULE_HH

#include "jetstream/module.hh"

namespace Jetstream::Modules {

struct PhaseCorrection : public Module::Config {
    F64 phaseIncrement = 0.0;

    JST_MODULE_TYPE(phase_correction);
    JST_MODULE_PARAMS(phaseIncrement);
};

}  // namespace Jetstream::Modules

#endif  // JETSTREAM_DOMAINS_DSP_PHASE_CORRECTION_MODULE_HH
