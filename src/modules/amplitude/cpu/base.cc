#include "../generic.cc"

#include "jetstream/backend/devices/cpu/helpers.hh"

namespace Jetstream {

template<Device D, typename IT, typename OT>
struct Amplitude<D, IT, OT>::Impl {
    F32 scalingCoeff = 0.0f;
    U64 numberOfElements = 0;
};

template<Device D, typename IT, typename OT>
Amplitude<D, IT, OT>::Amplitude() {
    pimpl = std::make_unique<Impl>();
}

template<Device D, typename IT, typename OT>
Amplitude<D, IT, OT>::~Amplitude() {
    pimpl.reset();
}

template<Device D, typename IT, typename OT>
Result Amplitude<D, IT, OT>::createCompute(const Context&) {
    JST_TRACE("Create Amplitude compute core using CPU backend.");
    return Result::SUCCESS;
}

template<>
Result Amplitude<Device::CPU, CF32, F32>::compute(const Context&) {
    for (U64 i = 0; i < input.buffer.size(); i++) {
        const auto& number = input.buffer[i];
        const auto& real = number.real();
        const auto& imag = number.imag();

        const auto& pwr = sqrtf((real * real) + (imag * imag));

        output.buffer[i] = 20.0f * Backend::ApproxLog10(pwr) + pimpl->scalingCoeff;
    }

    return Result::SUCCESS;
}

template<>
Result Amplitude<Device::CPU, F32, F32>::compute(const Context&) {
    for (U64 i = 0; i < input.buffer.size(); i++) {
        const auto& pwr = fabs(input.buffer[i]);
        output.buffer[i] = 20.0f * Backend::ApproxLog10(pwr) + pimpl->scalingCoeff;
    }

    return Result::SUCCESS;
}

JST_AMPLITUDE_CPU(JST_INSTANTIATION)
JST_AMPLITUDE_CPU(JST_BENCHMARK)

}  // namespace Jetstream
