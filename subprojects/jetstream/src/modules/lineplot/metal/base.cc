#include "jetstream/modules/lineplot.hh"

#include "../generic.cc"

namespace Jetstream {

template<Device D, typename T>
const Result Lineplot<D, T>::compute(const RuntimeMetadata& meta) {
    // TODO: Convert this to pure Metal.

    for (U64 i = 0; i < input.buffer.shape(1); i++) {
        plot[(i*3)+1] = 0.0;
    }

    for (U64 b = 0; b < input.buffer.shape(0); b++) {
        for (U64 i = 0; i < input.buffer.shape(1); i++) {
            plot[(i*3)+1] += input.buffer[{b, i}];
        }
    }

    // for (U64 i = 0; i < input.buffer.shape(1); i++) {
    //     plot[(i*3)+1] /= input.buffer.shape(0);
    // }

    return Result::SUCCESS;
}

template class Lineplot<Device::Metal, F32>;
    
}  // namespace Jetstream
