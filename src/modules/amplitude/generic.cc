#include "jetstream/modules/amplitude.hh"

#include "benchmark.cc"

namespace Jetstream {

template<Device D, typename IT, typename OT>
Result Amplitude<D, IT, OT>::create() {
    JST_DEBUG("Initializing Amplitude module.");
    JST_INIT_IO();

    // Calculate parameters.

    const U64 last_axis = input.buffer.rank() - 1;
    scalingCoeff = 20.0f * log10f(1.0f / input.buffer.shape()[last_axis]);

    // Allocate output.

    output.buffer = Tensor<D, OT>(input.buffer.shape());

    return Result::SUCCESS;
}

template<Device D, typename IT, typename OT>
void Amplitude<D, IT, OT>::info() const {
    JST_INFO("  None");
}

}  // namespace Jetstream
