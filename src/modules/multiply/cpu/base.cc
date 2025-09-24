#include "../generic.cc"

#include "jetstream/memory/devices/cpu/helpers.hh"
#include "jetstream/memory2/helpers.hh"

namespace Jetstream {

template<Device D, typename T>
struct Multiply<D, T>::Impl {
    mem2::Tensor a;
    mem2::Tensor b;
    mem2::Tensor c;
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
    mem2::AutomaticIterator([](const auto& a, const auto& b, auto& c) {
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
