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
    for (U64 i = 0; i < input.buffer.size(); i++) {
        output.buffer[i] = 20.0f * Backend::ApproxLog10(abs(input.buffer[i]) / scalingSize);
    }

    return Result::SUCCESS;
}

JST_AMPLITUDE_CPU(JST_INSTANTIATION);
    
}  // namespace Jetstream
