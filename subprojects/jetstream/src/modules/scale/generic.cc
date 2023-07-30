#include "jetstream/modules/scale.hh"

namespace Jetstream {

template<Device D, typename T>
Scale<D, T>::Scale(const Config& config,
                   const Input& input) 
         : config(config), input(input) {
    JST_DEBUG("Initializing Scale module.");
    
    // Initialize output.
    JST_CHECK_THROW(Module::initInput(this->input.buffer));
    JST_CHECK_THROW(Module::initOutput(this->output.buffer, this->input.buffer.shape()));
}

template<Device D, typename T>
void Scale<D, T>::summary() const {
    JST_INFO("  Amplitude (min, max): ({}, {})", config.range.min, config.range.max);
}
    
}  // namespace Jetstream
