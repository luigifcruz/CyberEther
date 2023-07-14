#include "../generic.cc"

namespace Jetstream {

template<Device D, typename T>
Result Constellation<D, T>::createCompute(const RuntimeMetadata&) {
    JST_TRACE("Create Constellation compute core using CPU backend.");

    timeSamples = Vector<Device::CPU, F32, 2>({config.viewSize.width, config.viewSize.height});
    decayFactor = pow(0.999, input.buffer.shape()[0]);

    return Result::SUCCESS;
}

template<Device D, typename T>
Result Constellation<D, T>::compute(const RuntimeMetadata&) {
    for (U64 x = 0; x < timeSamples.size(); x++) {
        timeSamples[x] *= decayFactor;
    }

    auto& v = input.buffer;

    float min_real = std::numeric_limits<float>::max();
    float max_real = std::numeric_limits<float>::min();
    float min_imag = std::numeric_limits<float>::max();
    float max_imag = std::numeric_limits<float>::min();

    for (const auto& value : v) {
        min_real = std::min(min_real, value.real());
        max_real = std::max(max_real, value.real());
        min_imag = std::min(min_imag, value.imag());
        max_imag = std::max(max_imag, value.imag());
    }

    for (U64 b = 0; b < input.buffer.shape()[0]; b++) {
        const auto offset = input.buffer.shapeToOffset({b, 0});

        for (U64 x = 0; x < input.buffer.shape()[1]; x++) {
            const CF32& sample = input.buffer[x + offset];

            const U64 r = ((sample.real() - min_real) / (max_real - min_real)) * timeSamples.shape()[0];
            const U64 i = ((sample.imag() - min_imag) / (max_imag - min_imag)) * timeSamples.shape()[0];

            if (r < timeSamples.shape()[0] and i < timeSamples.shape()[1]) {
                timeSamples[{r, i}] += 0.02;
            }
        }
    }

    return Result::SUCCESS;
}


template class Constellation<Device::CPU, CF32>;

}  // namespace Jetstream
