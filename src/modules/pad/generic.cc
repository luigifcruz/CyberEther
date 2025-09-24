#include "jetstream/modules/pad.hh"

namespace Jetstream {

template<Device D, typename T>
Result Pad<D, T>::create() {
    JST_DEBUG("Initializing Pad module.");
    JST_INIT_IO();

    // Check parameters.

    if (config.axis >= input.unpadded.rank()) {
        JST_ERROR("Configuration axis ({}) is larger than the input rank ({}).", config.axis,
                                                                                 input.unpadded.rank());
        return Result::ERROR;
    }

    // Calculate padded shape.

    mem2::Shape paddedShape = input.unpadded.shape();
    paddedShape[config.axis] += config.size;

    // Allocate output.

    JST_CHECK(output.padded.create(D, mem2::TypeToDataType<T>(), paddedShape));

    return Result::SUCCESS;
}

template<Device D, typename T>
void Pad<D, T>::info() const {
    JST_DEBUG("  Pad Size:   {}", config.size);
    JST_DEBUG("  Pad Axis:   {}", config.axis);
}

}  // namespace Jetstream
