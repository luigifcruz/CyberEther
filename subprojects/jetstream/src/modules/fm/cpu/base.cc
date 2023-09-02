#include "../generic.cc"

#include "jetstream/backend/devices/cpu/helpers.hh"

namespace Jetstream {

// TODO: Change algorithm to eliminate Atan2.
inline F32 phase(const CF32& c) {
    return Backend::ApproxAtan2(c.imag(), c.real());
}

template<Device D, typename T>
Result FM<D, T>::createCompute(const RuntimeMetadata&) {
    JST_TRACE("Create FM compute core.");
    return Result::SUCCESS;
}

template<Device D, typename T>
Result FM<D, T>::compute(const RuntimeMetadata&) {

    F32 prevPhase = phase(input.buffer[0]);
    for (U64 i = 1; i < input.buffer.size(); i++) {
        const F32 currPhase = phase(input.buffer[i]);
        F32 phaseDiff = currPhase - prevPhase;
        while (phaseDiff > M_PI) phaseDiff -= 2 * M_PI;
        while (phaseDiff < -M_PI) phaseDiff += 2 * M_PI;
        output.buffer[i-1] = phaseDiff;
        prevPhase = currPhase;
    }

    return Result::SUCCESS;
}

template class FM<Device::CPU, CF32>;
    
}  // namespace Jetstream
