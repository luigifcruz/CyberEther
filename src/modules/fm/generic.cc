#include "jetstream/modules/fm.hh"

namespace Jetstream {

template<Device D, typename T>
Result FM<D, T>::create() {
    JST_DEBUG("Initializing FM module.");

    // Initialize input/output.
    JST_INIT(
        JST_INIT_INPUT("buffer", input.buffer);
        JST_INIT_OUTPUT("buffer", output.buffer, input.buffer.shape());
    );

    // Initialize constant coefficients.

    kf = 100e3 / 240e3;
    ref = 1.0f / (2 * M_PI * kf);

    return Result::SUCCESS;
}

template<Device D, typename T>
void FM<D, T>::summary() const {
    JST_INFO("  Sample Rate: {:.2f} MHz", config.sampleRate / (1000*1000));
}

}  // namespace Jetstream
