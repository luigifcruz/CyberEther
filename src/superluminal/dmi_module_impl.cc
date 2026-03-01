#include "dmi_module_impl.hh"

namespace Jetstream::Modules {

Result DynamicTensorImportImpl::define() {
    JST_CHECK(defineInterfaceOutput("buffer"));

    return Result::SUCCESS;
}

Result DynamicTensorImportImpl::create() {
    JST_DEBUG("[SUPERLUMINAL] Initializing Dynamic Tensor Import module.");

    output = buffer;
    outputs()["buffer"] = {name(), "buffer", output};

    return Result::SUCCESS;
}

}  // namespace Jetstream::Modules
