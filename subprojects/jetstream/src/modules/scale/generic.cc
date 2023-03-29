#include "jetstream/modules/scale.hh"

namespace Jetstream {

template<Device D, typename T>
Scale<D, T>::Scale(const Config& config,
                   const Input& input) 
         : config(config), input(input) {
    JST_DEBUG("Initializing Scale module.");
    
    // Initialize output.
    JST_CHECK_THROW(this->initInput(this->input.buffer));
    JST_CHECK_THROW(this->initOutput(this->output.buffer, this->input.buffer.shape()));
}

template<Device D, typename T>
void Scale<D, T>::summary() const {
    JST_INFO("===== Scale Module Configuration");
    JST_INFO("Shape: {}", this->input.buffer.shape());
    JST_INFO("Amplitude (min, max): ({}, {})", config.range.min, config.range.max);
    JST_INFO("Input Type: {}", NumericTypeInfo<T>().name);
}
    
}  // namespace Jetstream
