#include "jetstream/modules/fft.hh"

namespace Jetstream {

template<Device D, typename T>
Result FFT<D, T>::create() {
    JST_DEBUG("Initializing FFT module.");

    std::vector<U64> outputShape = input.buffer.shape();
    if (config.offset != 0 || config.size != 0) {
        outputShape = {outputShape[0], config.size};
    }

    // Initialize Input & Output memory.
    JST_INIT(
        JST_INIT_INPUT("buffer", input.buffer);
        JST_INIT_OUTPUT("buffer", output.buffer, outputShape);
    );

    return Result::SUCCESS;
}

template<Device D, typename T>
void FFT<D, T>::summary() const {
    JST_INFO("  Forward: {}", config.forward ? "YES" : "NO");
    JST_INFO("  Offset: {}", config.offset);
    JST_INFO("  Size: {}", config.size);
}

}  // namespace Jetstream
