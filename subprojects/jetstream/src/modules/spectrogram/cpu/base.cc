#include "../generic.cc"

namespace Jetstream {

template<Device D, typename T>
const Result Spectrogram<D, T>::underlyingInitialize() {
    frequencyBins.resize(input.buffer.size() * config.viewSize.height);

    return Result::SUCCESS;
}

// 0: 1 1 1 1 1 1 1 1 1 1 1 1 1 1
// 1: 1 1 1 1 2 3 4 4 3 2 1 1 1 1 
// 2: 1 1 1 1 1 2 3 3 2 1 1 1 1 1
// 3: 1 1 1 1 1 1 2 2 1 1 1 1 1 1
// 4: 1 1 1 1 1 1 1 1 1 1 1 1 1 1
// |: | | | | | | | | | | | | | |
// 4: 0 0 0 0 0 0 1 1 0 0 0 0 0 0 
// 3: 0 0 0 0 0 1 1 1 1 0 0 0 0 0
// 2: 0 0 0 0 1 1 1 1 1 1 0 0 0 0
// 1: 5 5 5 5 4 3 2 2 3 4 5 5 5 5
// 0: 0 0 0 0 0 0 0 0 0 0 0 0 0 0

template<Device D, typename T>
const Result Spectrogram<D, T>::underlyingCompute(const RuntimeMetadata& meta) {
    for (U64 x = 0; x < input.buffer.size() * config.viewSize.height; x++) {
        frequencyBins[x] *= 0.999; 
    }

    for (U64 x = 0; x < input.buffer.size(); x++) {
        U16 index = input.buffer[x] * config.viewSize.height;

        if (index < config.viewSize.height && index > 0) {
            frequencyBins[x + (index * input.buffer.size())] += 0.02; 
        }
    }

    return Result::SUCCESS;
}

template class Spectrogram<Device::CPU, F64>;
template class Spectrogram<Device::CPU, F32>;

}  // namespace Jetstream
