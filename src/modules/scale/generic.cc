#include "jetstream/modules/scale.hh"

#include "benchmark.cc"

namespace Jetstream {

template<Device D, typename T>
Result Scale<D, T>::create() {
    JST_DEBUG("Initializing Scale module.");
    JST_INIT_IO();

    // Allocate output.

    output.buffer = Tensor<D, T>(input.buffer.shape());

    return Result::SUCCESS;
}

template<Device D, typename T>
void Scale<D, T>::info() const {
    JST_INFO("  Amplitude (min, max): ({}, {})", config.range.min, config.range.max);
}

}  // namespace Jetstream
