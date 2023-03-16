#include "jetstream/modules/fft.hh"

namespace Jetstream { 

template<Device D, typename T>
FFT<D, T>::FFT(const Config& config,
               const Input& input) 
         : config(config), input(input) {
    JST_DEBUG("Initializing FFT module.");

    JST_CHECK_THROW(this->initInput(this->input.buffer, {getBufferSize()}));
    JST_CHECK_THROW(this->initOutput(this->output.buffer, {getBufferSize()}));

    // Check parameters. 
    if (this->input.buffer.size() != this->config.size) {
        JST_FATAL("Input Buffer size ({}) is different than the" \
            "configuration size ({}).", this->input.buffer.size(),
            this->config.size);
        JST_CHECK_THROW(Result::ERROR);
    }
}

template<Device D, typename T>
void FFT<D, T>::summary() const {
    JST_INFO("===== FFT Module Configuration");
    JST_INFO("Size: {}", this->config.size);
    JST_INFO("Direction: {}", static_cast<I64>(config.direction));
    JST_INFO("Input Type: {}", NumericTypeInfo<T>().name);
}

}  // namespace Jetstream
