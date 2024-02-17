#include "../generic.cc"

namespace Jetstream {

template<Device D, typename T>
struct Lineplot<D, T>::Impl {
    std::vector<F32> sums;
};

template<Device D, typename T>
Lineplot<D, T>::Lineplot() {
    pimpl = std::make_unique<Impl>();
    gimpl = std::make_unique<GImpl>();
}

template<Device D, typename T>
Lineplot<D, T>::~Lineplot() {
    pimpl.reset();
    gimpl.reset();
}

template<Device D, typename T>
Result Lineplot<D, T>::createCompute(const Context&) {
    JST_TRACE("Create Multiply compute core using CPU backend.");

    pimpl->sums.resize(numberOfElements, 0.0f);

    return Result::SUCCESS;
}

template<Device D, typename T>
Result Lineplot<D, T>::compute(const Context&) {
    for (U64 i = 0; i < numberOfElements; ++i) {
        pimpl->sums[i] = 0.0f;
    }

    for (U64 b = 0; b < numberOfBatches; ++b) {
        for (U64 i = 0; i < numberOfElements; ++i) {
            pimpl->sums[i] += input.buffer[i + b * numberOfElements];
        }
    }

    for (U64 i = 0; i < numberOfElements; ++i) {
        plot[(i * 3) + 1] = (pimpl->sums[i] * normalizationFactor) - 1.0f;
    }

    return Result::SUCCESS;
}

JST_LINEPLOT_CPU(JST_INSTANTIATION)
JST_LINEPLOT_CPU(JST_BENCHMARK)

}  // namespace Jetstream