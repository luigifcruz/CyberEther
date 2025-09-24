#include "../generic.cc"

#include "jetstream/memory2/helpers.hh"

namespace Jetstream {

template<Device D, typename T>
struct Add<D, T>::Impl {
    mem2::Tensor a;
    mem2::Tensor b;
    mem2::Tensor c;
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
    mem2::AutomaticIterator([](const auto& a, const auto& b, auto& c) {
        c = a + b;
    }, impl->a, impl->b, impl->c);

    return Result::SUCCESS;
}

JST_ADD_CPU(JST_INSTANTIATION)

}  // namespace Jetstream
