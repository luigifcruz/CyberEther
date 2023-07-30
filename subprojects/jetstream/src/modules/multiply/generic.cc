#include "jetstream/modules/multiply.hh"

namespace Jetstream { 

template<Device D, typename T>
Multiply<D, T>::Multiply(const Config& config, 
                         const Input& input) 
         : config(config), input(input) {
    JST_DEBUG("Initializing Multiply module.");
    
    // Initialize output.
    JST_CHECK_THROW(Module::initInput(this->input.factorA));
    JST_CHECK_THROW(Module::initInput(this->input.factorB));
    JST_CHECK_THROW(Module::initOutput(this->output.product, this->input.factorA.shape()));

    // Check parameters.
    if (this->input.factorA.shape()[1] != this->input.factorB.shape()[1]) {
        JST_FATAL("Input A shape ({}) is different than the" \
            "Input B shape ({}).",
            this->input.factorA.shape(),
            this->input.factorB.shape());
        JST_CHECK_THROW(Result::ERROR);
    }
}

template<Device D, typename T>
void Multiply<D, T>::summary() const {
    JST_INFO("  None");
}

}  // namespace Jetstream
