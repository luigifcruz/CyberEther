#include "../generic.cc"

namespace Jetstream {

template<Device D, typename T>
const Result Spectrogram<D, T>::createCompute(const RuntimeMetadata& meta) {
    JST_TRACE("Create Spectrogram compute core using CPU backend.");

    frequencyBins = Vector<Device::CPU, F32, 2>({input.buffer.size(), config.viewSize.height});
    decayFactor = pow(0.999, input.buffer.shape(0));

    return Result::SUCCESS;
}

template<Device D, typename T>
const Result Spectrogram<D, T>::viewSizeCallback() {
    return Result::SUCCESS;
}

template<Device D, typename T>
const Result Spectrogram<D, T>::compute(const RuntimeMetadata& meta) {
    for (U64 x = 0; x < input.buffer.size() * config.viewSize.height; x++) {
        frequencyBins[x] *= decayFactor;
    }

    for (U64 b = 0; b < input.buffer.shape(0); b++) {
        const auto offset = input.buffer.shapeToOffset({b, 0});

        for (U64 x = 0; x < input.buffer.shape(1); x++) {
            const U16 index = input.buffer[x + offset] * config.viewSize.height;

            if (index < config.viewSize.height && index > 0) {
                frequencyBins[x + (index * input.buffer.shape(1))] += 0.02; 
            }
        }
    }

    return Result::SUCCESS;
}

template class Spectrogram<Device::CPU, F64>;
template class Spectrogram<Device::CPU, F32>;

}  // namespace Jetstream
