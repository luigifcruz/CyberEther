#include "../generic.cc"

#include "jetstream/memory/devices/cpu/helpers.hh"

namespace Jetstream {

template<Device D, typename T>
struct Multiply<D, T>::Impl {
    Tensor<D, T> a;
    Tensor<D, T> b;
    Tensor<D, T> c;
};

template<Device D, typename T>
Multiply<D, T>::Multiply() {
    impl = std::make_unique<Impl>();
}

template<Device D, typename T>
Multiply<D, T>::~Multiply() {
    impl.reset();
}

template<Device D, typename T>
Result Multiply<D, T>::createCompute(const Context&) {
    JST_TRACE("Create Multiply compute core using CPU backend.");
    return Result::SUCCESS;
}

template<Device D, typename T>
Result Multiply<D, T>::compute(const Context&) {
    Memory::CPU::AutomaticIterator([](const auto& a, const auto& b, auto& c) {
        if constexpr (std::is_same_v<T, CF32>) {
            c = std::complex<F32>(a.real() * b.real() - a.imag() * b.imag(),
                                  a.real() * b.imag() + a.imag() * b.real());
        } else {
            c = a * b;
        }
    }, impl->a, impl->b, impl->c);

    return Result::SUCCESS;
}

JST_MULTIPLY_CPU(JST_INSTANTIATION)
JST_MULTIPLY_CPU(JST_BENCHMARK)

}  // namespace Jetstream
