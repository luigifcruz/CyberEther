#include "../generic.cc"

namespace Jetstream {

template<Device D, typename T>
struct OverlapAdd<D, T>::Impl {
    Tensor<D, T> previousOverlap;
};

template<Device D, typename T>
OverlapAdd<D, T>::OverlapAdd() {
    impl = std::make_unique<Impl>();
}

template<Device D, typename T>
OverlapAdd<D, T>::~OverlapAdd() {
    impl.reset();
}

template<Device D, typename T>
Result OverlapAdd<D, T>::createCompute(const Context&) {
    JST_TRACE("Create Overlap Add compute core using CPU backend.");

    return Result::SUCCESS;
}

template<Device D, typename T>
Result OverlapAdd<D, T>::compute(const Context&) {
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
                sample += impl->previousOverlap[shape];
            } else {
                shape[0] -= 1;
                sample += input.overlap[shape];
            }
        }
    }

    // Get last batch element from overlap.

    {
        std::vector<U64> shape = impl->previousOverlap.shape();
        for (U64 i = 0; i < impl->previousOverlap.size(); i++) {
            impl->previousOverlap.offset_to_shape(i, shape);
            shape[0] = input.overlap.shape()[0] - 1;
            impl->previousOverlap[i] = input.overlap[shape];
        }
    }

    return Result::SUCCESS;
}

JST_OVERLAP_ADD_CPU(JST_INSTANTIATION)

}  // namespace Jetstream
