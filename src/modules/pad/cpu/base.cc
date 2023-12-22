#include "../generic.cc"

namespace Jetstream {

template<Device D, typename T>
Result Pad<D, T>::createCompute(const RuntimeMetadata&) {
    JST_TRACE("Create Pad compute core using CPU backend.");
    return Result::SUCCESS;
}

template<Device D, typename T>
Result Pad<D, T>::compute(const RuntimeMetadata&) {
    std::vector<U64> shape = input.unpadded.shape();

    for (U64 i = 0; i < input.unpadded.size(); i++) {
        input.unpadded.offset_to_shape(i, shape);
        output.padded[shape] = input.unpadded[i];
    }

    // TODO: Add offset.
    // TODO: Add blanking.

    return Result::SUCCESS;
}

JST_PAD_CPU(JST_INSTANTIATION);
    
}  // namespace Jetstream
