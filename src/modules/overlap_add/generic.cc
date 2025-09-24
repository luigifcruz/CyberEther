#include "jetstream/modules/overlap_add.hh"

namespace Jetstream {

template<Device D, typename T>
Result OverlapAdd<D, T>::create() {
    JST_DEBUG("Initializing Overlap Add module.");
    JST_INIT_IO();

    // Check parameters.

    if (config.axis >= input.buffer.rank()) {
        JST_ERROR("Configuration axis ({}) is larger than the input rank ({}).", config.axis,
                                                                                 input.buffer.rank());
        return Result::ERROR;
    }

    if (input.buffer.rank() != input.overlap.rank()) {
        JST_ERROR("Input buffer rank ({}) is not equal to the overlap rank ({}).",
                  input.buffer.rank(), input.overlap.rank());
        return Result::ERROR;
    }

    if (input.buffer.shape()[config.axis] < input.overlap.shape()[config.axis]) {
        JST_ERROR("Overlap buffer size ({}) is larger than the buffer size ({}).",
                  input.overlap.shape()[config.axis], input.buffer.shape()[config.axis]);
        return Result::ERROR;
    }

    // TODO: Add broadcasting support.

    // Allocate output.

    JST_CHECK(output.buffer.create(D, mem2::TypeToDataType<T>(), input.buffer.shape()));

    mem2::Shape previousOverlapShape = input.overlap.shape();
    if (input.buffer.rank() > 1) {
        previousOverlapShape[0] = 1;
    }
    JST_CHECK(impl->previousOverlap.create(D, mem2::TypeToDataType<T>(), previousOverlapShape));

    return Result::SUCCESS;
}

template<Device D, typename T>
void OverlapAdd<D, T>::info() const {
    JST_DEBUG("  Axis:   {}", config.axis);
}

}  // namespace Jetstream
