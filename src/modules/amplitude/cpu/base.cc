#include "../generic.cc"

#include "jetstream/backend/devices/cpu/helpers.hh"

namespace Jetstream {

template<Device D, typename IT, typename OT>
struct Amplitude<D, IT, OT>::Impl {};

template<Device D, typename IT, typename OT>
Amplitude<D, IT, OT>::Amplitude() {
    pimpl = std::make_unique<Impl>();
}

template<Device D, typename IT, typename OT>
Amplitude<D, IT, OT>::~Amplitude() {
    pimpl.reset();
}

template<Device D, typename IT, typename OT>
Result Amplitude<D, IT, OT>::createCompute(const RuntimeMetadata&) {
    JST_TRACE("Create Amplitude compute core using CPU backend.");
    return Result::SUCCESS;
}

template<Device D, typename IT, typename OT>
Result Amplitude<D, IT, OT>::compute(const RuntimeMetadata&) {    
    for (U64 i = 0; i < input.buffer.size(); i++) {
        const auto& number = input.buffer[i];
        const auto& real = number.real();
        const auto& imag = number.imag();

        const auto& pwr = sqrtf((real * real) + (imag * imag));

        output.buffer[i] = 20.0f * Backend::ApproxLog10(pwr) + scalingCoeff;
    }

    return Result::SUCCESS;
}

JST_AMPLITUDE_CPU(JST_INSTANTIATION)
JST_AMPLITUDE_CPU(JST_BENCHMARK)
    
}  // namespace Jetstream
