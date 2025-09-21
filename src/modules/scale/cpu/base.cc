#include "../generic.cc"

namespace Jetstream {

template<Device D, typename T>
struct Scale<D, T>::Impl {
    F32 scalingCoeff;
    F32 offsetCoeff;
    U64 numberOfElements;
};

template<Device D, typename T>
Scale<D, T>::Scale() {
    impl = std::make_unique<Impl>();
}

template<Device D, typename T>
Scale<D, T>::~Scale() {
    impl.reset();
}

template<Device D, typename T>
Result Scale<D, T>::createCompute(const Context&) {
    JST_TRACE("Create Scale compute core using CPU backend.");

    return Result::SUCCESS;
}

template<Device D, typename T>
Result Scale<D, T>::compute(const Context&) {
    for (U64 i = 0; i < input.buffer.size(); i++) {
        output.buffer[i] = input.buffer[i] * impl->scalingCoeff + impl->offsetCoeff;
    }

    return Result::SUCCESS;
}

JST_SCALE_CPU(JST_INSTANTIATION)
JST_SCALE_CPU(JST_BENCHMARK)

}  // namespace Jetstream
