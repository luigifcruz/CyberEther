#include "jetstream/modules/duplicate.hh"

#include "benchmark.cc"

namespace Jetstream {

template<Device D, typename T>
Result Duplicate<D, T>::create() {
    JST_DEBUG("Initializing Duplicate module.");
    JST_INIT_IO();

    // Allocate output.

    if constexpr (D == Device::CUDA) {
        JST_CHECK(output.buffer.create(D, mem2::TypeToDataType<T>(), input.buffer.shape(), config.hostAccessible));
    } else {
        JST_CHECK(output.buffer.create(D, mem2::TypeToDataType<T>(), input.buffer.shape()));
    }

    return Result::SUCCESS;
}

template<Device D, typename T>
void Duplicate<D, T>::info() const {
    JST_DEBUG("  Host Accessible: {}", config.hostAccessible);
}

}  // namespace Jetstream
