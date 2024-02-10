#include "jetstream/modules/duplicate.hh"

#include "benchmark.cc"

namespace Jetstream {

template<Device D, typename T>
Result Duplicate<D, T>::create() {
    JST_DEBUG("Initializing Duplicate module.");
    JST_INIT_IO();

    // Allocate output.

    output.buffer = Tensor<D, T>(input.buffer.shape());

    return Result::SUCCESS;
}

template<Device D, typename T>
void Duplicate<D, T>::info() const {
    JST_INFO("  None");
}

}  // namespace Jetstream
