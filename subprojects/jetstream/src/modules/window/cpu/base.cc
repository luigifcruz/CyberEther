#include "jetstream/modules/window.hh"

namespace Jetstream {

template<Device D, typename T>
Window<D, T>::Window(const Config& config, const Input& input) 
    : config(config), input(input) {
    JST_DEBUG("Initializing Window module with CPU backend.");

    // Intialize output.
    this->InitOutput(this->output.window, getWindowSize());

    // Generate FFT window.
    for (U64 i = 0; i < this->config.size; i++) {
        F64 tap;

        tap = 0.5 * (1 - cos(2 * M_PI * i / this->config.size));
        tap = (i % 2) == 0 ? tap : -tap;

        this->output.window[i] = T(tap, 0.0);
    }

    JST_INFO("===== Window Module Configuration");
    JST_INFO("Window Size: {}", this->config.size);
    JST_INFO("Output Type: {}", getTypeName<T>());
}

template class Window<Device::CPU, CF64>;
template class Window<Device::CPU, CF32>;
    
}  // namespace Jetstream
