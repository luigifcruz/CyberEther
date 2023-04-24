#include "jetstream/backend/base.hh"

namespace Jetstream::Backend {

template<>
const Result Instance::initialize<Device::CPU>(const Config& config) {
    if (!cpu) {
        JST_DEBUG("Initializing CPU backend.");
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
    if (!cpu) {
        JST_DEBUG("The CPU backend is not initialized.");
        JST_CHECK_THROW(Result::ERROR);
    }
    return cpu; 
}
  
}  // namespace Jetstream::Backend
