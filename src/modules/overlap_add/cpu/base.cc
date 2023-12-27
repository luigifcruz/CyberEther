#include "../generic.cc"

namespace Jetstream {

template<Device D, typename T>
Result OverlapAdd<D, T>::createCompute(const RuntimeMetadata&) {
    JST_TRACE("Create Overlap Add compute core using CPU backend.");

    return Result::SUCCESS;
}

template<Device D, typename T>
Result OverlapAdd<D, T>::compute(const RuntimeMetadata&) {
    // Copy input buffer to output buffer.

    for (U64 i = 0; i < input.buffer.size(); i++) {
        output.buffer[i] = input.buffer[i];
    }

    // Add overlap to output buffer.

    {
        std::vector<U64> shape = input.overlap.shape();
        for (U64 i = 0; i < input.overlap.size(); i++) {
            input.overlap.offset_to_shape(i, shape);
            auto& sample = output.buffer[shape];

            if (shape[0] == 0) {
                sample += previousOverlap[shape];
            } else {
                shape[0] -= 1;
                sample += input.overlap[shape];
            }
        }
    }
    
    // Get last batch element from overlap.

    {
        std::vector<U64> shape = previousOverlap.shape();
        for (U64 i = 0; i < previousOverlap.size(); i++) {
            previousOverlap.offset_to_shape(i, shape);
            shape[0] = input.overlap.shape()[0] - 1;
            previousOverlap[i] = input.overlap[shape];
        }
    }

    return Result::SUCCESS;
}

JST_OVERLAP_ADD_CPU(JST_INSTANTIATION)
    
}  // namespace Jetstream
