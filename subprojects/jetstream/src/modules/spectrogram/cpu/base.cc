#include "../generic.cc"

namespace Jetstream {

template<Device D, typename T>
const Result Spectrogram<D, T>::underlyingInitialize() {
    frequencyBins.resize(input.buffer.size() * config.viewSize.height);

    return Result::SUCCESS;
}

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
