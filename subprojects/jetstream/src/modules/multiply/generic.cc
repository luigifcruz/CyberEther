#include "jetstream/modules/multiply.hh"

namespace Jetstream { 

template<Device D, typename T>
Multiply<D, T>::Multiply(const Config& config, 
                         const Input& input) 
         : config(config), input(input) {
    JST_DEBUG("Initializing Multiply module.");
    
    // Initialize output.
    JST_CHECK_THROW(this->initInput(this->input.factorA, getBufferSize()));
    JST_CHECK_THROW(this->initInput(this->input.factorB, getBufferSize()));
    JST_CHECK_THROW(this->initOutput(this->output.product, getBufferSize()));

    // Check parameters. 
    if (this->input.factorA.size() != this->config.size) {
        JST_FATAL("Input A size ({}) is different than the" \
            "configuration size ({}).", this->input.factorA.size(),
            this->config.size);
        JST_CHECK_THROW(Result::ERROR);
    }

    if (this->input.factorB.size() != this->config.size) {
        JST_FATAL("Input B size ({}) is different than the" \
            "configuration size ({}).", this->input.factorB.size(), 
            this->config.size);
        JST_CHECK_THROW(Result::ERROR);
    }
}

template<Device D, typename T>
void Multiply<D, T>::summary() const {
    JST_INFO("===== Multiply Module Configuration");
    JST_INFO("Size: {}", this->config.size);
    JST_INFO("Input Type: {}", NumericTypeInfo<T>().name);
}

}  // namespace Jetstream
