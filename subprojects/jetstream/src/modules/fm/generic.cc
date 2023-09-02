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

    return Result::SUCCESS;
}

template<Device D, typename T>
void FM<D, T>::summary() const {
    JST_INFO("  None");
}

}  // namespace Jetstream
