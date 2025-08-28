#include "../generic.cc"

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

    gimpl->decayFactor = pow(0.999, input.buffer.shape()[0]);

    return Result::SUCCESS;
}

template<Device D, typename T>
Result Constellation<D, T>::compute(const Context&) {
    for (U64 x = 0; x < gimpl->timeSamples.size(); x++) {
        gimpl->timeSamples[x] *= gimpl->decayFactor;
    }

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

    for (U64 b = 0; b < input.buffer.shape()[0]; b++) {
        for (U64 x = 0; x < input.buffer.shape()[1]; x++) {
            const CF32& sample = input.buffer[{b, x}];

            const U64 r = ((sample.real() - min_real) / (max_real - min_real)) * gimpl->timeSamples.shape()[0];
            const U64 i = ((sample.imag() - min_imag) / (max_imag - min_imag)) * gimpl->timeSamples.shape()[0];

            if (r < gimpl->timeSamples.shape()[0] and i < gimpl->timeSamples.shape()[1]) {
                gimpl->timeSamples[{r, i}] += 0.02;
            }
        }
    }

    return Result::SUCCESS;
}

JST_CONSTELLATION_CPU(JST_INSTANTIATION)
JST_CONSTELLATION_CPU(JST_BENCHMARK)

}  // namespace Jetstream
