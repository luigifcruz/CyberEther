#include "jetstream/modules/amplitude.hh"

namespace Jetstream { 

template<Device D, typename IT, typename OT>
Amplitude<D, IT, OT>::Amplitude(const Config& config,
                                const Input& input)
         : config(config), input(input) {
    JST_DEBUG("Initializing Amplitude module.");
    
    // Initialize output.
    JST_CHECK_THROW(this->initInput(this->input.buffer));
    JST_CHECK_THROW(this->initOutput(this->output.buffer, this->input.buffer.shape()));
}

template<Device D, typename IT, typename OT>
void Amplitude<D, IT, OT>::summary() const {
    JST_INFO("===== Amplitude Module Configuration");
    JST_INFO("Shape: {}", this->input.buffer.shape());
    JST_INFO("Device: {}", DeviceTypeInfo<D>().name);
    JST_INFO("Input Type: {}", NumericTypeInfo<IT>().name);
    JST_INFO("Output Type: {}", NumericTypeInfo<OT>().name);
}

}  // namespace Jetstream
