#include "dmi_module_impl.hh"

namespace Jetstream::Modules {

Result DynamicTensorImportImpl::validate() {
    const auto& config = *candidate();

    if (!config.buffer.validShape()) {
        JST_ERROR("[MODULE_DYNAMIC_TENSOR_IMPORT] Buffer tensor is not initialized.");
        return Result::ERROR;
    }

    return Result::SUCCESS;
}

Result DynamicTensorImportImpl::define() {
    JST_CHECK(defineInterfaceOutput("buffer"));

    return Result::SUCCESS;
}

Result DynamicTensorImportImpl::create() {
    JST_DEBUG("[SUPERLUMINAL] Initializing Dynamic Tensor Import module.");

    output = buffer;
    outputs()["buffer"].produced(name(), "buffer", output);

    return Result::SUCCESS;
}

}  // namespace Jetstream::Modules
