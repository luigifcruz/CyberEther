#include "../generic.cc"

namespace Jetstream {

template<Device D, typename T>
Result Spectrogram<D, T>::createCompute(const RuntimeMetadata&) {
    JST_TRACE("Create Spectrogram compute core using CPU backend.");

    decayFactor = pow(0.999, numberOfBatches);

    return Result::SUCCESS;
}

template<Device D, typename T>
Result Spectrogram<D, T>::compute(const RuntimeMetadata&) {
    const U64& size = frequencyBins.size();
    const F32 factor = decayFactor;
    for (U64 x = 0; x < size; ++x) {
        frequencyBins[x] *= factor;
    }

    for (U64 b = 0; b < numberOfBatches; b++) {
        for (U64 x = 0; x < numberOfElements; x++) {
            const U16 index = input.buffer[{b, x}] * config.height;

            if (index < config.height && index > 0) {
                frequencyBins[x + (index * numberOfElements)] += 0.02; 
            }
        }
    }

    return Result::SUCCESS;
}

JST_SPECTROGRAM_CPU(JST_INSTANTIATION)

}  // namespace Jetstream
