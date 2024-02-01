#include "../generic.cc"

namespace Jetstream {

template<Device D, typename T>
struct Spectrogram<D, T>::Impl {};

template<Device D, typename T>
Spectrogram<D, T>::Spectrogram() {
    pimpl = std::make_unique<Impl>();
}

template<Device D, typename T>
Spectrogram<D, T>::~Spectrogram() {
    pimpl.reset();
}

template<Device D, typename T>
Result Spectrogram<D, T>::createCompute(const Context&) {
    JST_TRACE("Create Spectrogram compute core using CPU backend.");

    return Result::SUCCESS;
}

template<Device D, typename T>
Result Spectrogram<D, T>::compute(const Context&) {
    const U64& size = frequencyBins.size();
    const F32 factor = decayFactor;
    for (U64 x = 0; x < size; ++x) {
        frequencyBins[x] *= factor;
    }

    for (U64 b = 0; b < numberOfBatches; b++) {
        for (U64 x = 0; x < numberOfElements; x++) {
            const U16 index = input.buffer[{b, x}] * config.height;

            if (index < config.height && index > 0) {
                auto& val = frequencyBins[x + (index * numberOfElements)];
                val = std::min(val + 0.02, 1.0); 
            }
        }
    }

    return Result::SUCCESS;
}

JST_SPECTROGRAM_CPU(JST_INSTANTIATION)
JST_SPECTROGRAM_CPU(JST_BENCHMARK)

}  // namespace Jetstream
