#include "../generic.cc"

#include "jetstream/backend/devices/cpu/helpers.hh"

namespace Jetstream {

template<Device D, typename T>
Result Fold<D, T>::compute(const RuntimeMetadata&) {
    // Zero-out output buffer.

    for (U64 i = 0; i < output.buffer.size(); i++) {
        output.buffer[i] = 0.0f;
    }

    // Fold input buffer.

    std::vector<U64> shape = input.buffer.shape();
    for (U64 i = 0; i < input.buffer.size(); i++) {
        input.buffer.offset_to_shape(i, shape);

        // Add offset to axis.
        shape[config.axis] += config.offset;
        shape[config.axis] %= input.buffer.shape()[config.axis];

        // Fold.
        shape[config.axis] %= config.size;
        output.buffer[shape] += input.buffer[i];
    }

    // Average output buffer.

    for (U64 i = 0; i < output.buffer.size(); i++) {
        output.buffer[i] /= decimationFactor;
    }

    return Result::SUCCESS;
}

JST_FOLD_CPU(JST_INSTANTIATION);

}  // namespace Jetstream
