#include "../generic.cc"

namespace Jetstream {

template<Device D, typename T>
Result Unpad<D, T>::createCompute(const Context&) {
    JST_TRACE("Create Unpad compute core using CPU backend.");
    return Result::SUCCESS;
}

template<Device D, typename T>
Result Unpad<D, T>::compute(const Context&) {
    std::vector<U64> shape = input.padded.shape();
    const U64 pad_offset = shape[config.axis] - config.size;

    for (U64 i = 0; i < input.padded.size(); i++) {
        input.padded.offset_to_shape(i, shape);

        if (shape[config.axis] >= pad_offset) {
            shape[config.axis] -= pad_offset;
            output.pad[shape] = input.padded[i];
        } else {
            output.unpadded[shape] = input.padded[i];
        }
    }

    // TODO: Add offset.

    return Result::SUCCESS;
}

JST_UNPAD_CPU(JST_INSTANTIATION)
    
}  // namespace Jetstream
