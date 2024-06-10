#include "jetstream/modules/take.hh"

namespace Jetstream {

template<Device D, typename T>
Result Take<D, T>::create() {
    JST_DEBUG("Initializing Take module.");
    JST_INIT_IO();

    // Check parameters.

    if (config.axis >= input.buffer.rank()) {
        JST_ERROR("Configuration axis ({}) is larger than the input rank ({}).", config.axis,
                                                                                 input.buffer.rank());
        return Result::ERROR;
    }

    if (config.index >= input.buffer.shape()[config.axis]) {
        JST_ERROR("Configuration index ({}) is larger than the input shape ({}).", config.index,
                                                                                   input.buffer.shape()[config.axis]);
        return Result::ERROR;
    }

    // Calculate output shape.

    std::vector<U64> outputShape = input.buffer.shape();
    outputShape[config.axis] = 1;

    // Allocate output.

    output.buffer = Tensor<D, T>(outputShape);

    return Result::SUCCESS;
}

template<Device D, typename T>
void Take<D, T>::info() const {
    JST_DEBUG("  Index:   {}", config.index);
    JST_DEBUG("  Axis:   {}", config.axis);
}

}  // namespace Jetstream
