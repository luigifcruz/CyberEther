#include "jetstream/modules/fft.hh"

#include "benchmark.cc"

namespace Jetstream {

template<Device D, typename IT, typename OT>
Result FFT<D, IT, OT>::create() {
    JST_DEBUG("Initializing FFT module.");
    JST_INIT_IO();

    // Calculate parameters.
    
    // Determine which axis to use
    const U64 tensor_rank = input.buffer.rank();
    U64 fft_axis = (config.axis < 0) ? (tensor_rank - 1) : config.axis;
    
    // Ensure the axis is valid
    if (fft_axis >= tensor_rank) {
        JST_ERROR("[FFT] Invalid axis: {} (rank is {})", fft_axis, tensor_rank);
        return Result::ERROR;
    }

    numberOfElements = input.buffer.shape()[fft_axis];

    numberOfOperations = 1;
    for (U64 i = 0; i < tensor_rank; i++) {
        if (i != fft_axis) {
            numberOfOperations *= input.buffer.shape()[i];
        }
    }

    elementStride = 1;

    JST_TRACE("[FFT] Using axis: {}", fft_axis);
    JST_TRACE("[FFT] Number of elements: {}", numberOfElements);
    JST_TRACE("[FFT] Number of operations: {}", numberOfOperations);
    JST_TRACE("[FFT] Element stride: {}", elementStride);

    // Allocate output.

    output.buffer = Tensor<D, OT>(input.buffer.shape());

    return Result::SUCCESS;
}

template<Device D, typename IT, typename OT>
void FFT<D, IT, OT>::info() const {
    JST_DEBUG("  Forward: {}", config.forward ? "YES" : "NO");
    JST_DEBUG("  Axis: {}", (config.axis < 0) ? "Last axis" : std::to_string(config.axis));
}

}  // namespace Jetstream
