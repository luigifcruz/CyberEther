#include "jetstream/modules/unpad.hh"

namespace Jetstream {

template<Device D, typename T>
Result Unpad<D, T>::create() {
    JST_DEBUG("Initializing Unpad module.");
    JST_INIT_IO();

    // Check parameters.

    if (config.axis >= input.padded.rank()) {
        JST_ERROR("Configuration axis ({}) is larger than the input rank ({}).", config.axis,
                                                                                 input.padded.rank());
        return Result::ERROR;
    }

    if (config.size > input.padded.shape()[config.axis]) {
        JST_ERROR("Configuration size ({}) is larger than the input shape ({}).", config.size,
                                                                                  input.padded.shape());
        return Result::ERROR;
    }

    // Calculate padded shape.

    std::vector<U64> unpaddedShape = input.padded.shape();
    unpaddedShape[config.axis] -= config.size;

    // Allocate output.

    output.unpadded = Tensor<D, T>(unpaddedShape);

    unpaddedShape[config.axis] = config.size;
    output.pad = Tensor<D, T>(unpaddedShape);

    return Result::SUCCESS;
}

template<Device D, typename T>
void Unpad<D, T>::info() const {
    JST_INFO("  Pad Size:   {}", config.size);
    JST_INFO("  Pad Axis:   {}", config.axis);
}

}  // namespace Jetstream
