#include "jetstream/modules/invert.hh"

namespace Jetstream {

template<Device D, typename T>
Result Invert<D, T>::create() {
    JST_DEBUG("Initializing Invert module.");
    JST_INIT_IO();

    // Allocate output.

    output.buffer = Tensor<D, T>(input.buffer.shape());

    return Result::SUCCESS;
}

template<Device D, typename T>
void Invert<D, T>::info() const {
    JST_INFO("  None");
}

}  // namespace Jetstream
