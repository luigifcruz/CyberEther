#include "jetstream/backend/base.hh"

namespace Jetstream::Backend {

template<>
const Result Instance::initialize<Device::Vulkan>(const Config& config) {
    if (!vulkan) {
        JST_DEBUG("Initializing Vulkan backend.");
        vulkan = std::make_unique<Vulkan>(config);
    }
    return Result::SUCCESS;
}

template<>
const Result Instance::destroy<Device::Vulkan>() {
    if (vulkan) {
        vulkan.reset();
    }
    return Result::SUCCESS;
}

template<>
const auto& Instance::state<Device::Vulkan>() {
    if (!vulkan) {
        JST_DEBUG("The Vulkan backend is not initialized.");
        JST_CHECK_THROW(Result::ERROR);
    }
    return vulkan;
}
  
}  // namespace Jetstream::Backend
