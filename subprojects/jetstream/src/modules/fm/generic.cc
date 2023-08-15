#include "jetstream/modules/fm.hh"

namespace Jetstream { 

template<Device D, typename T>
FM<D, T>::FM(const Config& config, 
             const Input& input) 
         : config(config),
           input(input) {
    JST_DEBUG("Initializing FM module.");
    
    // Initialize input/output.
    JST_CHECK_THROW(Module::initInput(this->input.buffer));
    JST_CHECK_THROW(Module::initOutput(this->output.buffer, this->input.buffer.shape()));
}

template<Device D, typename T>
void FM<D, T>::summary() const {
    JST_INFO("  None");
}

}  // namespace Jetstream
