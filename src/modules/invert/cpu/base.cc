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
        auto value = input.buffer[i];

        if ((i % 2) == 0) {
            output.buffer[i] = {value.real(), value.imag()};
        } else {
            output.buffer[i] = {-value.real(), -value.imag()};
        }
    }

    return Result::SUCCESS;
}

JST_INVERT_CPU(JST_INSTANTIATION)
JST_INVERT_CPU(JST_BENCHMARK)
    
}  // namespace Jetstream
