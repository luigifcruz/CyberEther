#include "../generic.cc"

namespace Jetstream {

template<Device D, typename T>
RRCFilter<D, T>::RRCFilter() {
    impl = std::make_unique<Impl>();
}

template<Device D, typename T>
RRCFilter<D, T>::~RRCFilter() {
    impl.reset();
}

template<Device D, typename T>
Result RRCFilter<D, T>::compute(const Context&) {
    if (!impl->baked) {
        JST_CHECK(impl->generateRRCCoeffs(*this));
        impl->baked = true;
    }

    const U64 inputSize = input.buffer.size();
    const U64 numTaps = config.taps;

    for (U64 n = 0; n < inputSize; n++) {
        // Add current input sample to history
        impl->history[impl->historyIndex] = input.buffer[n];

        // Compute filter output using convolution
        T outputSample{};

        for (U64 k = 0; k < numTaps; k++) {
            const U64 histIdx = (impl->historyIndex + numTaps - k) % numTaps;

            if constexpr (std::is_same_v<T, CF32>) {
                outputSample += impl->history[histIdx] * std::complex<F32>(impl->coeffs[k], 0.0f);
            } else {
                outputSample += impl->history[histIdx] * impl->coeffs[k];
            }
        }

        output.buffer[n] = outputSample;

        // Update history index (circular buffer)
        impl->historyIndex = (impl->historyIndex + 1) % numTaps;
    }

    return Result::SUCCESS;
}

JST_RRC_FILTER_CPU(JST_INSTANTIATION)

}  // namespace Jetstream
