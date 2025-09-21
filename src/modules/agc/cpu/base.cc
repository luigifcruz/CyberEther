#include "../generic.cc"

#include "jetstream/backend/devices/cpu/helpers.hh"

namespace Jetstream {

template<Device D, typename T>
struct AGC<D, T>::Impl {};

template<Device D, typename T>
AGC<D, T>::AGC() {
    impl = std::make_unique<Impl>();
}

template<Device D, typename T>
AGC<D, T>::~AGC() {
    impl.reset();
}

template<Device D, typename T>
Result AGC<D, T>::compute(const Context&) {
    const F32 desiredLevel = 1.0f;

    // TODO: This is a dog shit implementation. Improve.

    F32 currentMax = 0.0f;
    for (U64 i = 0; i < input.buffer.size(); i++) {
        currentMax = std::max(currentMax, std::abs(input.buffer[i]));
    }

    const F32 gain = (currentMax != 0) ? (desiredLevel / currentMax) : 1.0f;

    for (U64 i = 0; i < output.buffer.size(); i++) {
        output.buffer[i] = input.buffer[i] * gain;
    }

    return Result::SUCCESS;
}

JST_AGC_CPU(JST_INSTANTIATION)
JST_AGC_CPU(JST_BENCHMARK)

}  // namespace Jetstream
