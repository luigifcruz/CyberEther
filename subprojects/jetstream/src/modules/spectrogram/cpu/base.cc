#include "../generic.cc"

namespace Jetstream {

template<Device D, typename T>
Result Spectrogram<D, T>::createCompute(const RuntimeMetadata&) {
    JST_TRACE("Create Spectrogram compute core using CPU backend.");

    frequencyBins = Vector<Device::CPU, F32, 2>({input.buffer.shape()[1], config.height});
    decayFactor = pow(0.999, input.buffer.shape()[0]);

    return Result::SUCCESS;
}

template<Device D, typename T>
Result Spectrogram<D, T>::compute(const RuntimeMetadata&) {
    const U64& size = frequencyBins.size();
    const F32 factor = decayFactor;
    for (U64 x = 0; x < size; ++x) {
        frequencyBins.at(x) *= factor;
    }

    for (U64 b = 0; b < input.buffer.shape()[0]; b++) {
        const auto offset = input.buffer.shapeToOffset({b, 0});

        for (U64 x = 0; x < input.buffer.shape()[1]; x++) {
            U16 index = input.buffer.at(x + offset) * config.height;

            if (index < config.height && index > 0) {
               frequencyBins.at(x + (index * input.buffer.shape()[1])) += 0.02; 
            }
        }
    }

    return Result::SUCCESS;
}

// TODO: Remove in favor of module manifest.
template class Spectrogram<Device::CPU, F64>;
template class Spectrogram<Device::CPU, F32>;

}  // namespace Jetstream
