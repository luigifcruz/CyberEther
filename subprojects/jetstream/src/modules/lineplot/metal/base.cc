#include "jetstream/modules/lineplot.hh"

#include "../generic.cc"

namespace Jetstream {

template<Device D, typename T>
const Result Lineplot<D, T>::compute(const RuntimeMetadata& meta) {
    // TODO: Convert this to pure Metal. Without a kernel if possible.
    for (size_t i = 0; i < getBufferSize(); i++) {
        plot[(i*3)+1] = input.buffer[i];
    }
    return Result::SUCCESS;
}

template class Lineplot<Device::Metal, F32>;
    
}  // namespace Jetstream
