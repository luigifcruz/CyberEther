#include "../generic.cc"

#include "jetstream/backend/devices/cpu/helpers.hh"

namespace Jetstream {

template<Device D, typename T>
Result FM<D, T>::createCompute(const RuntimeMetadata&) {
    JST_TRACE("Create FM compute core.");
    return Result::SUCCESS;
}

template<Device D, typename T>
Result FM<D, T>::compute(const RuntimeMetadata&) {
    for (size_t n = 1; n < input.buffer.size(); n++) {
        output.buffer[n] = std::arg(std::conj(input.buffer[n - 1]) * input.buffer[n]) * ref;
    }

    return Result::SUCCESS;
}

template class FM<Device::CPU, CF32>;
    
}  // namespace Jetstream
