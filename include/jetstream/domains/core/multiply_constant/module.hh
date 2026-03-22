#ifndef JETSTREAM_DOMAINS_CORE_MULTIPLY_CONSTANT_MODULE_HH
#define JETSTREAM_DOMAINS_CORE_MULTIPLY_CONSTANT_MODULE_HH

#include "jetstream/module.hh"
#include "jetstream/types.hh"

namespace Jetstream::Modules {

struct MultiplyConstant : public Module::Config {
    F32 constant = 1.0f;

    JST_MODULE_TYPE(multiply_constant);
    JST_MODULE_PARAMS(constant);
};

}  // namespace Jetstream::Modules

#endif  // JETSTREAM_DOMAINS_CORE_MULTIPLY_CONSTANT_MODULE_HH
