#include "jetstream/modules/amplitude.hh"

namespace Jetstream { 

template<Device D, typename IT, typename OT>
Amplitude<D, IT, OT>::Amplitude(const Config& config,
                                const Input& input)
         : config(config), input(input) {
    JST_DEBUG("Initializing Amplitude module.");
    
    // Initialize output.
    JST_CHECK_THROW(this->initInput(this->input.buffer, {getBufferSize()}));
    JST_CHECK_THROW(this->initOutput(this->output.buffer, {getBufferSize()}));

    // Check parameters. 
    if (this->input.buffer.size() != this->config.size) {
        JST_FATAL("Input buffer size ({}) is different than the" \
            "configuration size ({}).", this->input.buffer.size(), 
            this->config.size);
        JST_CHECK_THROW(Result::ERROR);
    }
}

template<Device D, typename IT, typename OT>
void Amplitude<D, IT, OT>::summary() const {
    JST_INFO("===== Amplitude Module Configuration");
    JST_INFO("Size: {}", this->config.size);
    JST_INFO("Device: {}", DeviceTypeInfo<D>().name);
    JST_INFO("Input Type: {}", NumericTypeInfo<IT>().name);
    JST_INFO("Output Type: {}", NumericTypeInfo<OT>().name);
}

}  // namespace Jetstream
