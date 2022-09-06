#include "../generic.cc"

namespace Jetstream {

template<Device D, typename T>
const Result Spectrogram<D, T>::underlyingInitialize() {
    intermediate.resize(input.buffer.size() * config.height);
    frequencyBins.resize(input.buffer.size() * config.height);

    return Result::SUCCESS;
}

// 0: 1 1 1 1 1 1 1 1 1 1 1 1 1 1
// 1: 1 1 1 1 2 3 4 4 3 2 1 1 1 1 
// 2: 1 1 1 1 1 2 3 3 2 1 1 1 1 1
// 3: 1 1 1 1 1 1 2 2 1 1 1 1 1 1
// 4: 1 1 1 1 1 1 1 1 1 1 1 1 1 1
// C: | | | | | | | | | | | | | |
// 4: 0 0 0 0 0 0 1 1 0 0 0 0 0 0 
// 3: 0 0 0 0 0 1 1 1 1 0 0 0 0 0
// 2: 0 0 0 0 1 1 1 1 1 1 0 0 0 0
// 1: 5 5 5 5 4 3 2 2 3 4 5 5 5 5
// 0: 0 0 0 0 0 0 0 0 0 0 0 0 0 0
template<typename T>
static inline T scale(const T x, const T min, const T max) {
    return (x - min) / (max - min);
}

template<Device D, typename T>
const Result Spectrogram<D, T>::underlyingCompute(const RuntimeMetadata& meta) {
    std::copy(input.buffer.data(), input.buffer.data() + input.buffer.size(), 
              intermediate.begin() + (inc * input.buffer.size()));

    for (U64 i = 0; i < 2048 * 512; i++) {
        frequencyBins[i] = 0.0; 
    }

    for (U64 x = 0; x < 2048; x++) {
        for (U64 y = 0; y < 512; y++) {
            auto index = intermediate[x + (y * 2048)] * 512.0;
            if (index < 512 && index > 0.0) {
                frequencyBins[x + (U64(index) * 2048)] += 1.0; 
            }
        }
    }

    auto nmax = *std::max_element(frequencyBins.begin(), frequencyBins.end());
    auto nmin = *std::min_element(frequencyBins.begin(), frequencyBins.end());

    for (U64 i = 0; i < 2048 * 512; i++) {
        frequencyBins[i] = (frequencyBins[i] - nmin) * (1.0 / (nmax - nmin));

        if (frequencyBins[i] > 0.3) {
            frequencyBins[i] = 0.0;
        }
    }

    nmax = *std::max_element(frequencyBins.begin(), frequencyBins.end());
    nmin = *std::min_element(frequencyBins.begin(), frequencyBins.end());

    for (U64 i = 0; i < 2048 * 512; i++) {
        frequencyBins[i] = (frequencyBins[i] - nmin) * (1.0 / (nmax - nmin));
    }

    return Result::SUCCESS;
}

template class Spectrogram<Device::CPU, F64>;
template class Spectrogram<Device::CPU, F32>;

}  // namespace Jetstream
