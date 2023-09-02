#include "jetstream/modules/multiply_constant.hh"

namespace Jetstream {

template<Device D, typename T>
Result MultiplyConstant<D, T>::create() {
    JST_DEBUG("Initializing Multiply Constant module.");

    // Initialize output.
    JST_INIT(
        JST_INIT_INPUT("factor", input.factor);
        JST_INIT_OUTPUT("product", output.product, input.factor.shape());
    );

    return Result::SUCCESS;
}

template<Device D, typename T>
void MultiplyConstant<D, T>::summary() const {
    // TODO: Add custom formater for complex type.
    if constexpr (IsComplex<T>::value) {
        JST_INFO("  Constant: ({}, {})", config.constant.real(), config.constant.imag());
    } else {
        JST_INFO("  Constant: {}", config.constant);
    }
}

}  // namespace Jetstream
