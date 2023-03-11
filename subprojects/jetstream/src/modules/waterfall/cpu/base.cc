#include "../generic.cc"

namespace Jetstream {

template<Device D, typename T>
const Result Waterfall<D, T>::createCompute(const RuntimeMetadata& meta) {
    frequencyBins = Vector<Device::CPU, F32>(input.buffer.size() * config.height);

    return Result::SUCCESS;
}

template<Device D, typename T>
const Result Waterfall<D, T>::underlyingCompute() {
    std::copy(input.buffer.data(), input.buffer.data() + input.buffer.size(), 
              frequencyBins.begin() + (inc * input.buffer.size()));

    return Result::SUCCESS;
}

template class Waterfall<Device::CPU, F64>;
template class Waterfall<Device::CPU, F32>;
    
}  // namespace Jetstream
