#include "../generic.cc"

#include "jetstream/backend/devices/cpu/helpers.hh"

namespace Jetstream {

template<Device D, typename T>
Result Invert<D, T>::createCompute(const RuntimeMetadata&) {
    JST_TRACE("Create Invert compute core.");
    return Result::SUCCESS;
}

template<Device D, typename T>
Result Invert<D, T>::compute(const RuntimeMetadata&) {
    for (U64 i = 0; i < input.buffer.size(); i++) {
        output.buffer[i] = {input.buffer[i].real(), input.buffer[i].imag()};
        output.buffer[i] *= (i % 2) == 0 ? CF32{1.0f, 0.0f} : CF32{-1.0f, 0.0f};
    }

    return Result::SUCCESS;
}

template class Invert<Device::CPU, CF32>;
    
}  // namespace Jetstream
