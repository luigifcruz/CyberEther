#include "jetstream/modules/window.hh"

namespace Jetstream {

template<Device D, typename T>
Result Window<D, T>::create() {
    JST_DEBUG("Initializing Window module.");
    JST_INIT_IO();

    // Allocate output.
    output.window = Tensor<D, T>({config.size});

    // Configure initial state.
    baked = false;

    return Result::SUCCESS;
}

template<Device D, typename T>
Result Window<D, T>::compute(const RuntimeMetadata&) {
    if (baked) {
        return Result::SUCCESS;
    }

    // Generate FFT window.

    for (U64 i = 0; i < output.window.size(); i++) {
        F64 tap = 0.42 - 0.50 * std::cos(2.0 * M_PI * i / (output.window.size() - 1)) + \
                  0.08 * std::cos(4.0 * M_PI * i / (output.window.size() - 1));
        output.window.cpu()[i] = T(tap, 0.0);
    }

    baked = true;

    return Result::SUCCESS;
}

template<Device D, typename T>
void Window<D, T>::info() const {
    JST_INFO("  Window Size: {}", config.size);
}

JST_WINDOW_CPU(JST_INSTANTIATION);

}  // namespace Jetstream
