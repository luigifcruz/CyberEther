#include "../generic.cc"

namespace Jetstream {

template<Device D, typename T>
Result Multiply<D, T>::createCompute(const RuntimeMetadata&) {
    JST_TRACE("Create Multiply compute core using CPU backend.");
    return Result::SUCCESS;
}

template<Device D, typename T>
Result Multiply<D, T>::compute(const RuntimeMetadata&) {
    std::vector<U64> shape_p(output.product.rank(), 1);

    for (U64 idx = 0; idx < output.product.size(); idx++) {
        output.product.offset_to_shape(idx, shape_p);
        output.product[idx] = input.factorA[shape_p] * input.factorB[shape_p];
    }

    return Result::SUCCESS;
}

JST_MULTIPLY_CPU(JST_INSTANTIATION);
    
}  // namespace Jetstream
