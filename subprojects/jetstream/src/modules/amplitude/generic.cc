#include "jetstream/modules/amplitude.hh"

namespace Jetstream {

template<Device D, typename IT, typename OT>
Result Amplitude<D, IT, OT>::create() {
    JST_DEBUG("Initializing Amplitude module.");

    // Initialize output.
    JST_INIT(
        JST_INIT_INPUT("buffer", input.buffer);
        JST_INIT_OUTPUT("buffer", output.buffer, input.buffer.shape());
    );

    return Result::SUCCESS;
}

template<Device D, typename IT, typename OT>
void Amplitude<D, IT, OT>::summary() const {
    JST_INFO("  None");
}

}  // namespace Jetstream
