#include "../generic.cc"

#include "jetstream/backend/devices/cpu/helpers.hh"

namespace Jetstream {

template<Device D, typename IT, typename OT>
Result Amplitude<D, IT, OT>::createCompute(const RuntimeMetadata&) {
    JST_TRACE("Create Amplitude compute core using CPU backend.");
    return Result::SUCCESS;
}

template<Device D, typename IT, typename OT>
Result Amplitude<D, IT, OT>::compute(const RuntimeMetadata&) {
    const auto& fftSize = input.buffer.shape()[1];

    for (U64 i = 0; i < input.buffer.size(); i++) {
        output.buffer[i] = 20.0f * Backend::ApproxLog10(abs(input.buffer[i]) / fftSize);
    }

    return Result::SUCCESS;
}

template class Amplitude<Device::CPU, CF32>;
    
}  // namespace Jetstream
