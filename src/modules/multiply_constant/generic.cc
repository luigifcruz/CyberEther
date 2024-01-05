#include "jetstream/modules/multiply_constant.hh"

#include "benchmark.cc"

namespace Jetstream {

template<Device D, typename T>
Result MultiplyConstant<D, T>::create() {
    JST_DEBUG("Initializing Multiply Constant module.");
    JST_INIT_IO();

    // Allocate output.

    output.product = Tensor<D, T>(input.factor.shape());

    return Result::SUCCESS;
}

template<Device D, typename T>
void MultiplyConstant<D, T>::info() const {
    // TODO: Add custom formater for complex type.
    if constexpr (IsComplex<T>::value) {
        JST_INFO("  Constant: ({}, {})", config.constant.real(), config.constant.imag());
    } else {
        JST_INFO("  Constant: {}", config.constant);
    }
}

}  // namespace Jetstream
