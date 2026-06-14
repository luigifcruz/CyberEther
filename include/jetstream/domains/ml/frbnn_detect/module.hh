#ifndef JETSTREAM_DOMAINS_ML_FRBNN_DETECT_MODULE_HH
#define JETSTREAM_DOMAINS_ML_FRBNN_DETECT_MODULE_HH

#include "jetstream/memory/types.hh"
#include "jetstream/module.hh"

namespace Jetstream::Modules {

struct FrbnnDetect : public Module::Config {
    F32 threshold  = 0.5f;
    U64 classIndex = 0;

    JST_MODULE_TYPE(frbnn_detect);
    JST_MODULE_PARAMS(threshold, classIndex);
};

}  // namespace Jetstream::Modules

#endif  // JETSTREAM_DOMAINS_ML_FRBNN_DETECT_MODULE_HH
