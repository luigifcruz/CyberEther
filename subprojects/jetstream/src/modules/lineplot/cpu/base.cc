#include "jetstream/modules/lineplot.hh"

#include "../generic.cc"

namespace Jetstream {

template<Device D, typename T>
Result Lineplot<D, T>::underlyingCreateCompute(const RuntimeMetadata&) {
    JST_TRACE("Create Multiply compute core using CPU backend.");

    return Result::SUCCESS;
}

template<Device D, typename T>
Result Lineplot<D, T>::compute(const RuntimeMetadata&) {
    const U64 num_batches = input.buffer.shape(0);
    const U64 num_samples = input.buffer.shape(1);

    for (U64 i = 0; i < num_samples; ++i) {
        F32 sum = 0.0;

        for (U64 b = 0; b < num_batches; ++b) {
            sum += input.buffer[{b, i}];
        }

        plot[{i, 1}] = (sum / (0.5f * num_batches)) - 1.0f;
    }

    return Result::SUCCESS;
}

template class Lineplot<Device::CPU, F64>;
template class Lineplot<Device::CPU, F32>;
    
}  // namespace Jetstream
