#include "jetstream/modules/lineplot.hh"

#include "../generic.cc"

namespace Jetstream {

template<Device D, typename T>
const Result Lineplot<D, T>::compute(const RuntimeMetadata& meta) {
    for (size_t i = 0; i < getBufferSize(); i++) {
        plot[(i*3)+1] = input.buffer[i];
    }
    return Result::SUCCESS;
}

template class Lineplot<Device::CPU, F64>;
template class Lineplot<Device::CPU, F32>;
    
}  // namespace Jetstream
