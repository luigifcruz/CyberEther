#include "module_impl.hh"

namespace Jetstream::Modules {

Result DuplicateImpl::validate() {
    const auto& config = *candidate();

    if (config.outputDevice.empty()) {
        JST_ERROR("[DUPLICATE] Output device is not specified.");
        return Result::ERROR;
    }

    if (StringToDevice(config.outputDevice) == DeviceType::None) {
        JST_ERROR("[DUPLICATE] Invalid output device: {}.", config.outputDevice);
        return Result::ERROR;
    }

    return Result::SUCCESS;
}

Result DuplicateImpl::define() {
    JST_CHECK(defineTaint(Module::Taint::DISCONTIGUOUS));

    JST_CHECK(defineInterfaceInput("buffer"));
    JST_CHECK(defineInterfaceOutput("buffer"));

    return Result::SUCCESS;
}

Result DuplicateImpl::create() {
    // Setup input buffer.

    input = inputs().at("buffer").tensor;

    // Setup output buffer.

    Buffer::Config outputConfig{};
    outputConfig.hostAccessible = hostAccessible;
    JST_CHECK(output.create(StringToDevice(outputDevice), input.dtype(), input.shape(), outputConfig));
    JST_CHECK(output.propagateAttributes(input));

    outputs()["buffer"].produced(name(), "buffer", output);

    // Setup staging buffer.

    if (output.device() == input.device()) {
        staging = output;
        return Result::SUCCESS;
    }

    if (output.hasDevice(input.device())) {
        staging.create(input.device(), output);
        return Result::SUCCESS;
    }

    JST_ERROR("[DUPLICATE] Cannot copy tensor from DeviceType::{} to DeviceType::{}.", input.device(),
                                                                                       StringToDevice(outputDevice));
    return Result::ERROR;
}

}  // namespace Jetstream::Modules
