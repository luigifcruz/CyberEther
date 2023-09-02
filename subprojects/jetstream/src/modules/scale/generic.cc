#include "jetstream/modules/scale.hh"

namespace Jetstream {

template<Device D, typename T>
Result Scale<D, T>::create() {
    JST_DEBUG("Initializing Scale module.");

    // Initialize output.
    JST_INIT(
        JST_INIT_INPUT("buffer", input.buffer);
        JST_INIT_OUTPUT("buffer", output.buffer, input.buffer.shape());
    );

    return Result::SUCCESS;
}

template<Device D, typename T>
void Scale<D, T>::summary() const {
    JST_INFO("  Amplitude (min, max): ({}, {})", config.range.min, config.range.max);
}

}  // namespace Jetstream
