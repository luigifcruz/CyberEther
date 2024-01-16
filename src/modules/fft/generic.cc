#include "jetstream/modules/fft.hh"

#include "benchmark.cc"

namespace Jetstream {

template<Device D, typename IT, typename OT>
Result FFT<D, IT, OT>::create() {
    JST_DEBUG("Initializing FFT module.");
    JST_INIT_IO();

    // Calculate parameters.

    const U64 last_axis = input.buffer.rank() - 1;

    numberOfElements = input.buffer.shape()[last_axis];

    numberOfOperations = 1;
    for (U64 i = 0; i < last_axis; i++) {
        numberOfOperations *= input.buffer.shape()[i];
    }

    elementStride = 1;

    JST_TRACE("[FFT] Number of elements: {};", numberOfElements);
    JST_TRACE("[FFT] Number of operations: {};", numberOfOperations);
    JST_TRACE("[FFT] Element stride: {};", elementStride);

    // TODO: Implement axis selection for FFT.

    // Allocate output.

    output.buffer = Tensor<D, OT>(input.buffer.shape());

    return Result::SUCCESS;
}

template<Device D, typename IT, typename OT>
void FFT<D, IT, OT>::info() const {
    JST_INFO("  Forward: {}", config.forward ? "YES" : "NO");
}

}  // namespace Jetstream
