#include "jetstream/modules/throttle.hh"

namespace Jetstream {

template<Device D, typename T>
struct Throttle<D, T>::Impl {
    std::chrono::steady_clock::time_point lastExecutionTime;
};

template<Device D, typename T>
Throttle<D, T>::Throttle() {
    impl = std::make_unique<Impl>();
}

template<Device D, typename T>
Throttle<D, T>::~Throttle() {
    impl.reset();
}

template<Device D, typename T>
Result Throttle<D, T>::create() {
    JST_DEBUG("Initializing Throttle module.");
    JST_INIT_IO();

    // Initialize timing.

    intervalMs(config.intervalMs);

    // Install bypass.

    output.buffer = input.buffer;

    return Result::SUCCESS;
}

template<Device D, typename T>
const U64& Throttle<D, T>::intervalMs(const U64& intervalMs) {
    // Update configuration.

    config.intervalMs = intervalMs;

    // Reset timing to allow immediate first pass.

    impl->lastExecutionTime = std::chrono::steady_clock::now() -
                              std::chrono::milliseconds(config.intervalMs);

    return config.intervalMs;
}

template<Device D, typename T>
void Throttle<D, T>::info() const {
    JST_DEBUG("  Throttle interval: {} ms", config.intervalMs);
}

template<Device D, typename T>
Result Throttle<D, T>::computeReady() {
    auto now = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        now - impl->lastExecutionTime);

    // Sleep until the configured interval has elapsed.
    if (elapsed < std::chrono::milliseconds(config.intervalMs)) {
        std::this_thread::sleep_for(std::chrono::milliseconds(config.intervalMs) - elapsed);
    }

    return Result::SUCCESS;
}

template<Device D, typename T>
Result Throttle<D, T>::compute(const Context&) {
    auto now = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        now - impl->lastExecutionTime);

    if (elapsed < std::chrono::milliseconds(config.intervalMs)) {
        return Result::YIELD;
    }

    // Update the timestamp for the next delay interval.
    impl->lastExecutionTime = now;

    return Result::SUCCESS;
}

JST_THROTTLE_CPU(JST_INSTANTIATION)

}  // namespace Jetstream
