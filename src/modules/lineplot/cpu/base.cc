#include "jetstream/modules/lineplot.hh"

#include "../generic.cc"

namespace Jetstream {

template<Device D, typename T>
Result Lineplot<D, T>::createCompute(const RuntimeMetadata&) {
    JST_TRACE("Create Multiply compute core using CPU backend.");

    return Result::SUCCESS;
}

template<Device D, typename T>
Result Lineplot<D, T>::compute(const RuntimeMetadata&) {
    for (U64 i = 0; i < numberOfElements; ++i) {
        F32 sum = 0.0;

        for (U64 b = 0; b < numberOfBatches; ++b) {
            sum += input.buffer[{b, i}];
        }

        plot[{i, 1}] = (sum / (0.5f * numberOfBatches)) - 1.0f;
    }

    return Result::SUCCESS;
}

JST_LINEPLOT_CPU(JST_INSTANTIATION);

}  // namespace Jetstream