#include "jetstream/modules/agc.hh"

namespace Jetstream {

template<Device D, typename T>
Result AGC<D, T>::create() {
    JST_DEBUG("Initializing AGC module.");
    JST_INIT_IO();

    // Allocate output.

    output.buffer = Tensor<D, T>(input.buffer.shape());

    return Result::SUCCESS;
}

template<Device D, typename T>
void AGC<D, T>::info() const {
    JST_INFO("  None");
}

}  // namespace Jetstream
