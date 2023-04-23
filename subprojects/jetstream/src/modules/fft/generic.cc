#include "jetstream/modules/fft.hh"

namespace Jetstream { 

template<Device D, typename T>
FFT<D, T>::FFT(const Config& config,
               const Input& input) 
         : config(config), input(input) {
    JST_DEBUG("Initializing FFT module.");

    JST_CHECK_THROW(this->initInput(this->input.buffer));
    JST_CHECK_THROW(this->initOutput(this->output.buffer, this->input.buffer.shape()));
}

template<Device D, typename T>
void FFT<D, T>::summary() const {
    JST_INFO("    Direction: {}", static_cast<I64>(config.direction));
}

}  // namespace Jetstream
