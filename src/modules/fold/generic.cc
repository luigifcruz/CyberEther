#include "jetstream/modules/fold.hh"

namespace Jetstream {

template<Device D, typename T>
Result Fold<D, T>::create() {
    JST_DEBUG("Initializing Fold module.");
    JST_INIT_IO();

    // Check parameters.

    if (input.buffer.rank() <= config.axis) {
        JST_ERROR("Axis configuration parameter ({}) is out of bounds.", config.axis);
        return Result::ERROR;
    }

    if (input.buffer.shape()[config.axis] % config.size != 0) {
        JST_ERROR("Size configuration parameter ({}) is not a divisor of the input buffer's shape ({}) along the axis ({}).",
                  config.size, input.buffer.shape()[config.axis], config.axis);
        return Result::ERROR;
    }

    if (input.buffer.shape()[config.axis] < config.offset) {
        JST_ERROR("Offset configuration parameter ({}) is greater than the input buffer's shape ({}) along the axis ({}).",
                  config.offset, input.buffer.shape()[config.axis], config.axis);
        return Result::ERROR;
    }

    // Calculate parameters.

    mem2::Shape output_shape = input.buffer.shape();
    output_shape[config.axis] = config.size;

    impl->decimationFactor = input.buffer.shape()[config.axis] / config.size;

    // Allocate output.

    JST_CHECK(output.buffer.create(D, mem2::TypeToDataType<T>(), output_shape));

    return Result::SUCCESS;
}

template<Device D, typename T>
void Fold<D, T>::info() const {
    JST_DEBUG("  Axis:   {}", config.axis);
    JST_DEBUG("  Size:   {}", config.size);
    JST_DEBUG("  Offset: {}", config.offset);
}

}  // namespace Jetstream
