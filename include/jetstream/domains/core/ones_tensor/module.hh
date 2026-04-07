#ifndef JETSTREAM_DOMAINS_CORE_ONES_TENSOR_MODULE_HH
#define JETSTREAM_DOMAINS_CORE_ONES_TENSOR_MODULE_HH

#include <string>
#include <vector>

#include "jetstream/module.hh"

namespace Jetstream::Modules {

struct OnesTensor : public Module::Config {
    std::vector<U64> shape = {1};
    std::string dataType = "F32";

    JST_MODULE_TYPE(ones_tensor);
    JST_MODULE_PARAMS(shape, dataType);
};

}  // namespace Jetstream::Modules

#endif  // JETSTREAM_DOMAINS_CORE_ONES_TENSOR_MODULE_HH
