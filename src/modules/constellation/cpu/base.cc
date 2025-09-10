#include "../generic.cc"

#include "jetstream/memory/devices/cpu/helpers.hh"

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

    // Use the total number of samples for decay calculation
    U64 sampleCount = input.buffer.size();
    gimpl->decayFactor = pow(0.999, sampleCount);

    return Result::SUCCESS;
}

template<Device D, typename T>
Result Constellation<D, T>::compute(const Context&) {
    Memory::CPU::AutomaticIterator([&](auto& sample) {
        sample *= gimpl->decayFactor;
    }, gimpl->timeSamples);

    auto& v = input.buffer;

    float min_real = std::numeric_limits<float>::max();
    float max_real = std::numeric_limits<float>::min();
    float min_imag = std::numeric_limits<float>::max();
    float max_imag = std::numeric_limits<float>::min();

    for (const auto& value : v) {
        min_real = std::min(min_real, value.real());
        max_real = std::max(max_real, value.real());
        min_imag = std::min(min_imag, value.imag());
        max_imag = std::max(max_imag, value.imag());
    }

    // Avoid division by zero
    float real_range = max_real - min_real;
    float imag_range = max_imag - min_imag;

    if (real_range == 0.0f) real_range = 1.0f;
    if (imag_range == 0.0f) imag_range = 1.0f;

    Memory::CPU::AutomaticIterator([&](const auto& sample) {
        const U64 r = ((sample.real() - min_real) / real_range) * (gimpl->timeSamples.shape()[0] - 1);
        const U64 i = ((sample.imag() - min_imag) / imag_range) * (gimpl->timeSamples.shape()[1] - 1);

        if (r < gimpl->timeSamples.shape()[0] && i < gimpl->timeSamples.shape()[1]) {
            gimpl->timeSamples[{r, i}] += 0.75;
        }
    }, input.buffer);

    return Result::SUCCESS;
}

JST_CONSTELLATION_CPU(JST_INSTANTIATION)
JST_CONSTELLATION_CPU(JST_BENCHMARK)

}  // namespace Jetstream
