#include "jetstream/backend/base.hh"

namespace Jetstream::Backend {

template<>
const Result Instance::initialize<Device::Vulkan>(const Config& config) {
    if (!vulkan) {
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
    return vulkan;
}
  
}  // namespace Jetstream::Backend
