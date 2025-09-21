#include "../generic.cc"

#include "jetstream/memory/devices/cpu/helpers.hh"

namespace Jetstream {

template<Device D, typename T>
struct Add<D, T>::Impl {
    Tensor<D, T> a;
    Tensor<D, T> b;
    Tensor<D, T> c;
};

template<Device D, typename T>
Add<D, T>::Add() {
    impl = std::make_unique<Impl>();
}

template<Device D, typename T>
Add<D, T>::~Add() {
    impl.reset();
}

template<Device D, typename T>
Result Add<D, T>::createCompute(const Context&) {
    JST_TRACE("Create Add compute core using CPU backend.");
    return Result::SUCCESS;
}

template<Device D, typename T>
Result Add<D, T>::compute(const Context&) {
    Memory::CPU::AutomaticIterator([](const auto& a, const auto& b, auto& c) {
        c = a + b;
    }, impl->a, impl->b, impl->c);

    return Result::SUCCESS;
}

JST_ADD_CPU(JST_INSTANTIATION)

}  // namespace Jetstream
