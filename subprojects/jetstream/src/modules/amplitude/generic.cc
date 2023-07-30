#include "jetstream/modules/amplitude.hh"

namespace Jetstream { 

template<Device D, typename IT, typename OT>
Amplitude<D, IT, OT>::Amplitude(const Config& config,
                                const Input& input)
         : config(config), input(input) {
    JST_DEBUG("Initializing Amplitude module.");
    
    // Initialize output.
    JST_CHECK_THROW(Module::initInput(this->input.buffer));
    JST_CHECK_THROW(Module::initOutput(this->output.buffer, this->input.buffer.shape()));
}

template<Device D, typename IT, typename OT>
void Amplitude<D, IT, OT>::summary() const {
    JST_INFO("  None");
}

}  // namespace Jetstream
