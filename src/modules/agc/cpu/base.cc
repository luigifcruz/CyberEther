#include "../generic.cc"

#include "jetstream/backend/devices/cpu/helpers.hh"
#include "jetstream/memory2/helpers.hh"

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

    mem2::View<const T> inputView(input.buffer);
    mem2::View<T> outputView(output.buffer);

    F32 currentMax = 0.0f;
    mem2::AutomaticIterator([&currentMax](const auto& val) {
        currentMax = std::max(currentMax, std::abs(val));
    }, input.buffer);

    const F32 gain = (currentMax != 0) ? (desiredLevel / currentMax) : 1.0f;

    mem2::AutomaticIterator([gain](const auto& in_val, auto& out_val) {
        out_val = in_val * gain;
    }, input.buffer, output.buffer);

    return Result::SUCCESS;
}

JST_AGC_CPU(JST_INSTANTIATION)
JST_AGC_CPU(JST_BENCHMARK)

}  // namespace Jetstream
