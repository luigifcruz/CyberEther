#include "jetstream/modules/invert.hh"

namespace Jetstream {

template<Device D, typename T>
Result Invert<D, T>::create() {
    JST_DEBUG("Initializing Invert module.");

    // Initialize input/output.
    JST_INIT(
        JST_INIT_INPUT("buffer", input.buffer);
        JST_INIT_OUTPUT("buffer", output.buffer, input.buffer.shape());
    );

    return Result::SUCCESS;
}

template<Device D, typename T>
void Invert<D, T>::summary() const {
    JST_INFO("  None");
}

}  // namespace Jetstream
