#include "jetstream/modules/scale.hh"

namespace Jetstream {

template<Device D, typename T>
Scale<D, T>::Scale(const Config& config,
                   const Input& input) 
         : config(config), input(input) {
    JST_DEBUG("Initializing Scale module.");
    
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

template<Device D, typename T>
void Scale<D, T>::summary() const {
    JST_INFO("===== Scale Module Configuration");
    JST_INFO("Size: {}", this->config.size);
    JST_INFO("Amplitude (min, max): ({}, {})", config.range.min, config.range.max);
    JST_INFO("Input Type: {}", NumericTypeInfo<T>().name);
}
    
}  // namespace Jetstream
