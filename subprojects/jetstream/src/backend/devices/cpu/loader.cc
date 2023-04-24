#include "jetstream/backend/base.hh"

namespace Jetstream::Backend {

template<>
const Result Instance::initialize<Device::CPU>(const Config& config) {
    if (!cpu) {
        cpu = std::make_unique<CPU>(config);
    }
    return Result::SUCCESS;
}

template<>
const Result Instance::destroy<Device::CPU>() {
    if (cpu) {
        cpu.reset();
    }
    return Result::SUCCESS;
}

template<>
const auto& Instance::state<Device::CPU>() {
    return cpu; 
}
  
}  // namespace Jetstream::Backend
