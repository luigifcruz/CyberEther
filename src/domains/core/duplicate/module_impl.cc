#include "module_impl.hh"

#include <exception>

namespace Jetstream::Modules {

namespace {

bool HasBufferBackend(const DeviceType device) {
    switch (device) {
        case DeviceType::CPU:
            return true;
        case DeviceType::CUDA:
#ifdef JETSTREAM_BACKEND_CUDA_AVAILABLE
            return true;
#else
            return false;
#endif
        case DeviceType::Metal:
#ifdef JETSTREAM_BACKEND_METAL_AVAILABLE
            return true;
#else
            return false;
#endif
        case DeviceType::Vulkan:
#ifdef JETSTREAM_BACKEND_VULKAN_AVAILABLE
            return true;
#else
            return false;
#endif
        case DeviceType::None:
        case DeviceType::WebGPU:
            return false;
    }

    return false;
}

}  // namespace

Result DuplicateImpl::validate() {
    const auto& config = *candidate();

    if (config.outputDevice.empty()) {
        JST_ERROR("[DUPLICATE] Output device is not specified.");
        return Result::ERROR;
    }

    if (!IsDeviceName(config.outputDevice)) {
        JST_ERROR("[DUPLICATE] Invalid output device: {}.", config.outputDevice);
        return Result::ERROR;
    }

    const auto configuredTargetDevice = StringToDevice(config.outputDevice);
    const auto candidateTargetDevice = (configuredTargetDevice == DeviceType::None)
        ? device()
        : configuredTargetDevice;

    if (!HasBufferBackend(candidateTargetDevice)) {
        JST_ERROR("[DUPLICATE] Output device {} has no buffer backend in this build.",
                  candidateTargetDevice);
        return Result::ERROR;
    }

    validatedTargetDevice = candidateTargetDevice;
    return Result::SUCCESS;
}

Result DuplicateImpl::define() {
    JST_CHECK(defineTaint(Module::Taint::DISCONTIGUOUS | Module::Taint::STATELESS));

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
    JST_CHECK(output.create(validatedTargetDevice,
                            input.dtype(),
                            input.shape(),
                            outputConfig));
    JST_CHECK(output.propagateAttributes(input));

    // Setup staging buffer.

    try {
        if (output.device() == input.device()) {
            staging = output;
        } else {
            JST_CHECK(staging.create(input.device(), output));
        }
    } catch (const std::exception& e) {
        JST_ERROR("[DUPLICATE] Failed to construct staging buffer: {}.", e.what());
        return Result::ERROR;
    } catch (const Result result) {
        return result;
    }

    outputs()["buffer"].produced(name(), "buffer", output);
    return Result::SUCCESS;
}

}  // namespace Jetstream::Modules
