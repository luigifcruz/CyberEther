#include "../generic.cc"

#include "jetstream/backend/devices/cpu/helpers.hh"

namespace Jetstream {

template<Device D, typename T>
struct Take<D, T>::Impl {};

template<Device D, typename T>
Take<D, T>::Take() {
    impl = std::make_unique<Impl>();
}

template<Device D, typename T>
Take<D, T>::~Take() {
    impl.reset();
}

template<Device D, typename T>
Result Take<D, T>::compute(const Context&) {
    std::vector<U64> shape = input.buffer.shape();
    for (U64 i = 0; i < output.buffer.size(); i++) {
        output.buffer.offset_to_shape(i, shape);
        shape[config.axis] = config.index;
        output.buffer[i] = input.buffer[shape];
    }

    return Result::SUCCESS;
}

JST_TAKE_CPU(JST_INSTANTIATION)

}  // namespace Jetstream
