#include "jetstream/modules/scale.hh"

#include "benchmark.cc"

namespace Jetstream {

template<Device D, typename T>
Result Scale<D, T>::create() {
    JST_DEBUG("Initializing Scale module.");
    JST_INIT_IO();

    // Calculate parameters.

    impl->numberOfElements = input.buffer.size();

    // Initialize coefficients.

    range(config.range);

    // Allocate output.

    output.buffer = Tensor<D, T>(input.buffer.shape());

    return Result::SUCCESS;
}

template<Device D, typename T>
const Range<T>& Scale<D, T>::range(const Range<T>& range) {
    // Update configuration.

    config.range = range;

    // Calculate parameters.

    impl->scalingCoeff = 1.0f / (config.range.max - config.range.min);
    impl->offsetCoeff = -config.range.min * impl->scalingCoeff;

    return range;
}

template<Device D, typename T>
void Scale<D, T>::info() const {
    JST_DEBUG("  Amplitude (min, max): ({}, {})", config.range.min, config.range.max);
}

}  // namespace Jetstream
