#include "jetstream/modules/multiply_constant.hh"

namespace Jetstream { 

template<Device D, typename T>
MultiplyConstant<D, T>::MultiplyConstant(const Config& config, 
                                         const Input& input) 
         : config(config), input(input) {
    JST_DEBUG("Initializing Multiply Constant module.");
    
    // Initialize output.
    JST_CHECK_THROW(Module::initInput(this->input.factor));
    JST_CHECK_THROW(Module::initOutput(this->output.product, this->input.factor.shape()));
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
