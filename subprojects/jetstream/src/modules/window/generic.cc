#include "jetstream/modules/window.hh"

namespace Jetstream {

template<Device D, typename T>
Window<D, T>::Window(const Config& config, 
                     const Input& input)
         : config(config), input(input) {
    JST_DEBUG("Initializing Window module.");

    // Initialize output.
    JST_CHECK_THROW(this->initOutput(this->output.window, config.shape));

    // Generate FFT window.
    for (U64 b = 0; b < config.shape[0]; b++) {
        for (U64 i = 0; i < config.shape[1]; i++) {
            F64 tap;

            tap = 0.5 * (1 - cos(2 * M_PI * i / config.shape[1]));
            tap = (i % 2) == 0 ? tap : -tap;

            this->output.window[{b, i}] = T(tap, 0.0);
        }
    }
}

template<Device D, typename T>
void Window<D, T>::summary() const {
    JST_INFO("    Window Shape: {}", this->config.shape);
}

}  // namespace Jetstream
