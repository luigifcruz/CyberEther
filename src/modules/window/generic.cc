#include "jetstream/modules/window.hh"

namespace Jetstream {

template<Device D, typename T>
struct Window<D, T>::Impl {
    bool baked = false;
};

template<Device D, typename T>
Window<D, T>::Window() {
    impl = std::make_unique<Impl>();
}

template<Device D, typename T>
Window<D, T>::~Window() {
    impl.reset();
}

template<Device D, typename T>
Result Window<D, T>::create() {
    JST_DEBUG("Initializing Window module.");
    JST_INIT_IO();

    // Allocate output.
    JST_CHECK(output.window.create(D, mem2::TypeToDataType<T>(), {config.size}));

    // Configure initial state.
    impl->baked = false;

    return Result::SUCCESS;
}

template<Device D, typename T>
Result Window<D, T>::compute(const Context&) {
    if (impl->baked) {
        return Result::SUCCESS;
    }

    // Generate FFT window.

    mem2::View<T> window(output.window);

    for (U64 i = 0; i < output.window.size(); i++) {
        F64 tap = 0.42 - 0.50 * std::cos(2.0 * JST_PI * i / (output.window.size() - 1)) + \
                  0.08 * std::cos(4.0 * JST_PI * i / (output.window.size() - 1));
        window[{i}] = T(tap, 0.0);
    }

    impl->baked = true;

    return Result::SUCCESS;
}

template<Device D, typename T>
void Window<D, T>::info() const {
    JST_DEBUG("  Window Size: {}", config.size);
}

JST_WINDOW_CPU(JST_INSTANTIATION)

}  // namespace Jetstream
