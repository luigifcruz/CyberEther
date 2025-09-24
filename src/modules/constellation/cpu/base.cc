#include "../generic.cc"
#include "jetstream/memory2/helpers.hh"
#include "jetstream/types.hh"

namespace Jetstream {

template<Device D, typename T>
struct Constellation<D, T>::Impl {
};

template<Device D, typename T>
Constellation<D, T>::Constellation() {
    pimpl = std::make_unique<Impl>();
    gimpl = std::make_unique<GImpl>();
}

template<Device D, typename T>
Constellation<D, T>::~Constellation() {
    pimpl.reset();
    gimpl.reset();
}

template<Device D, typename T>
Result Constellation<D, T>::createCompute(const Context&) {
    JST_TRACE("Create Constellation compute core using CPU backend.");

    if (input.buffer.size() == 0) {
        JST_ERROR("Input buffer is empty in createCompute.");
        return Result::ERROR;
    }

    return Result::SUCCESS;
}

template<Device D, typename T>
Result Constellation<D, T>::compute(const Context&) {
    // Get position buffer from components.
    std::span<Extent2D<F32>> positions;
    JST_CHECK(gimpl->shapes->getPositions("constellation_points", positions));

    // Update positions buffer.
    for (U64 i = 0; i < input.buffer.size(); i++) {
        const auto complexValue = input.buffer[i];
        positions[i] = { complexValue.real(), complexValue.imag() };
    }

    // Commit changes to the positions buffer.
    JST_CHECK(gimpl->shapes->updatePositions());

    return Result::SUCCESS;
}

JST_CONSTELLATION_CPU(JST_INSTANTIATION)
JST_CONSTELLATION_CPU(JST_BENCHMARK)

}  // namespace Jetstream
