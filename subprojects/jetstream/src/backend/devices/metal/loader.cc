#include "jetstream/backend/base.hh"

namespace Jetstream::Backend {

template<>
const Result Instance::initialize<Device::Metal>(const Config& config) {
    if (!metal) {
        metal = std::make_unique<Metal>(config);
    }
    return Result::SUCCESS;
}

template<>
const Result Instance::destroy<Device::Metal>() {
    if (metal) {
        metal.reset();
    }
    return Result::SUCCESS;
}

template<>
const auto& Instance::state<Device::Metal>() {
    return metal; 
}
  
}  // namespace Jetstream::Backend
