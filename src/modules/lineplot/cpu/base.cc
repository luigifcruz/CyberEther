#include "../generic.cc"

namespace Jetstream {

template<Device D, typename T>
Result Lineplot<D, T>::createCompute(const RuntimeMetadata&) {
    JST_TRACE("Create Multiply compute core using CPU backend.");

    return Result::SUCCESS;
}

template<Device D, typename T>
Result Lineplot<D, T>::compute(const RuntimeMetadata&) {
    const F32 normalizationFactor = 1.0f / (0.5f * numberOfBatches);
    std::vector<F32> sums(numberOfElements, 0.0f);

    for (U64 b = 0; b < numberOfBatches; ++b) {
        for (U64 i = 0; i < numberOfElements; ++i) {
            sums[i] += input.buffer[i + b * numberOfElements];
        }
    }

    for (U64 i = 0; i < numberOfElements; ++i) {
        plot[(i * 3) + 1] = (sums[i] * normalizationFactor) - 1.0f;
    }

    return Result::SUCCESS;
}

JST_LINEPLOT_CPU(JST_INSTANTIATION)
JST_LINEPLOT_CPU(JST_BENCHMARK)

}  // namespace Jetstream