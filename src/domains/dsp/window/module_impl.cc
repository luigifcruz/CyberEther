#include "module_impl.hh"

namespace Jetstream::Modules {

Result WindowImpl::validate() {
    const auto& config = *candidate();

    if (config.size == 0) {
        JST_ERROR("[MODULE_WINDOW] Window size cannot be zero.");
        return Result::ERROR;
    }

    return Result::SUCCESS;
}

Result WindowImpl::define() {
    JST_CHECK(defineInterfaceOutput("window"));

    return Result::SUCCESS;
}

Result WindowImpl::create() {
    // Allocate output tensor.
    JST_CHECK(output.create(device(), DataType::CF32, {size}));

    outputs()["window"].produced(name(), "window", output);

    // Configure initial state.
    baked = false;

    return Result::SUCCESS;
}

}  // namespace Jetstream::Modules
