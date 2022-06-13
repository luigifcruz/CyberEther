#include "jetstream/modules/lineplot.hh"

#include "../generic.cc"

namespace Jetstream {

template<Device D, typename T>
const Result Waterfall<D, T>::underlyingInitialize() {
    JST_INFO("Device init")

    frequencyBins.resize(input.buffer.size() * config.size.height);

    return Result::SUCCESS;
}

template<Device D, typename T>
const Result Waterfall<D, T>::underlyingCompute(const RuntimeMetadata& meta) {
    std::copy(input.buffer.data(), input.buffer.data() + input.buffer.size(), 
              frequencyBins.begin() + (inc * input.buffer.size()));

    return Result::SUCCESS;
}

template class Waterfall<Device::CPU, F64>;
template class Waterfall<Device::CPU, F32>;
    
}  // namespace Jetstream
