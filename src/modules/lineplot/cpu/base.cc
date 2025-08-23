#include "../generic.cc"

#include "jetstream/backend/devices/cpu/helpers.hh"

namespace Jetstream {

template<Device D, typename T>
struct Lineplot<D, T>::Impl {
    Tensor<Device::CPU, F32> sums;
    Tensor<Device::CPU, F32> averaging;
    Tensor<Device::CPU, F32> line;
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

    pimpl->sums = Tensor<Device::CPU, F32>({numberOfElements});
    pimpl->averaging = Tensor<Device::CPU, F32>({numberOfElements});

    for (U64 i = 0; i < numberOfElements; i++) {
        signalPoints[(i * 2) + 0] = i * 2.0f / (numberOfElements - 1) - 1.0f;
        signalPoints[(i * 2) + 1] = 0.0f;
    }

    return Result::SUCCESS;
}

template<Device D, typename T>
Result Lineplot<D, T>::compute(const Context&) {
    for (U64 i = 0; i < numberOfElements; i++) {
        pimpl->sums[i] = 0.0f;
    }

    for (U64 b = 0; b < numberOfBatches; b++) {
        for (U64 i = 0; i < numberOfElements; i++) {
            pimpl->sums[i] += input.buffer[(i * config.decimation) + b * numberOfElements];
        }
    }

    for (U64 i = 0; i < numberOfElements; i++) {
        // Get amplitude
        const auto& amplitude = (pimpl->sums[i] * normalizationFactor) - 1.0f;

        // Calculate moving average
        auto& average = pimpl->averaging[i];
        average -= average / config.averaging;
        average += amplitude / config.averaging;

        signalPoints[(i * 2) + 1] = average;
    }

    updateSignalPointsFlag = true;

    return Result::SUCCESS;
}

JST_LINEPLOT_CPU(JST_INSTANTIATION)
JST_LINEPLOT_CPU(JST_BENCHMARK)

}  // namespace Jetstream