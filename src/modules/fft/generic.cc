#include "jetstream/modules/fft.hh"

#include "benchmark.cc"

namespace Jetstream {

template<Device D, typename IT, typename OT>
Result FFT<D, IT, OT>::create() {
    JST_DEBUG("Initializing FFT module.");
    JST_INIT_IO();

    // Calculate parameters.

    const U64 last_axis = input.buffer.rank() - 1;

    pimpl->numberOfElements = input.buffer.shape()[last_axis];

    pimpl->numberOfOperations = 1;
    for (U64 i = 0; i < last_axis; i++) {
        pimpl->numberOfOperations *= input.buffer.shape()[i];
    }

    pimpl->elementStride = 1;

    JST_TRACE("[FFT] Number of elements: {};", pimpl->numberOfElements);
    JST_TRACE("[FFT] Number of operations: {};", pimpl->numberOfOperations);
    JST_TRACE("[FFT] Element stride: {};", pimpl->elementStride);

    // TODO: Implement axis selection for FFT.

    // Allocate output.

    JST_CHECK(output.buffer.create(D, mem2::TypeToDataType<OT>(), input.buffer.shape()));

    return Result::SUCCESS;
}

template<Device D, typename IT, typename OT>
void FFT<D, IT, OT>::info() const {
    JST_DEBUG("  Forward: {}", config.forward ? "YES" : "NO");
}

}  // namespace Jetstream
