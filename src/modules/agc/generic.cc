#include "jetstream/modules/agc.hh"

#include "benchmark.cc"

namespace Jetstream {

template<Device D, typename T>
Result AGC<D, T>::create() {
    JST_DEBUG("Initializing AGC module.");
    JST_INIT_IO();

    // Allocate output.

    JST_CHECK(output.buffer.create(D, mem2::TypeToDataType<T>(), input.buffer.shape()));

    return Result::SUCCESS;
}

template<Device D, typename T>
void AGC<D, T>::info() const {
    JST_DEBUG("  None");
}

}  // namespace Jetstream
