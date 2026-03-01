#ifndef JETSTREAM_SUPERLUMINAL_DMI_MODULE_HH
#define JETSTREAM_SUPERLUMINAL_DMI_MODULE_HH

#include "jetstream/module.hh"

namespace Jetstream::Modules {

struct DynamicTensorImport : public Module::Config {
    Tensor buffer;

    JST_MODULE_TYPE(dynamic_tensor_import);
    JST_MODULE_PARAMS(buffer);
};

}  // namespace Jetstream::Modules

#endif  // JETSTREAM_SUPERLUMINAL_DMI_MODULE_HH
